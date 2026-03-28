import rasterio
import numpy as np
import csv
import pandas as pd
from rasterio.windows import Window
from rasterio.crs import CRS
import pyproj
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')


# 定义安全除法函数，避免除零和NaN错误
def safe_divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            (denominator == 0) | np.isnan(denominator) | np.isnan(numerator),
            np.nan,
            numerator / denominator
        )
    return result


# --- 1. 核心参数设置（适配你的Excel路径+列名） ---
base_path = r'C:\Users\Che\Desktop\today\1-5'  # TIF文件基础路径
excel_path = r'C:\Users\Che\Desktop\arkall.xlsx'  # 你的Excel文件路径
output_csv = fr'{base_path}\rtk_points_values_with_indices.csv'  # 输出CSV路径

# TIFF文件路径（保持原有配置）
tiff_files = [
    fr'{base_path}\result_Green_veg.tif',
    fr'{base_path}\result_Red_veg.tif',
    fr'{base_path}\result_RedEdge_veg.tif',
    fr'{base_path}\result_NIR_veg.tif'
]

# 配置参数（保持原有逻辑）
SQUARE_SIZE = 30  # 30x30像素窗口
CUSTOM_NODATA = None  # 用影像自带NoData
CALCULATE_STD = True  # 计算标准差
CALCULATE_VALID_PIXELS = True  # 计算有效像素数
RTK_CRS = "EPSG:4326"  # Excel中经纬度是WGS84
band_names = ["Green", "Red", "Red_Edge", "NIR"]  # 与tiff_files顺序对应

# 5个区域配置（保持原有逻辑）
area_config = [
    ("中心", 0, 0, 1),
    ("上方", -SQUARE_SIZE, 0, 2),
    ("下方", SQUARE_SIZE, 0, 3),
    ("左侧", 0, -SQUARE_SIZE, 4),
    ("右侧", 0, SQUARE_SIZE, 5)
]

# 植被指数计算字典（保持原有逻辑）
vegetation_indices = {
    # 基础指数
    'NDVI': lambda rr: safe_divide(rr['NIR_Mean'] - rr['Red_Mean'], rr['NIR_Mean'] + rr['Red_Mean']),
    'RCI': lambda rr: safe_divide(rr['Red_Edge_Mean'], rr['Red_Mean']),
    'PSSR': lambda rr: safe_divide(rr['NIR_Mean'], rr['Red_Mean']),
    'SR': lambda rr: safe_divide(rr['NIR_Mean'], rr['Red_Mean']),
    'CI_green': lambda rr: safe_divide(rr['NIR_Mean'], rr['Green_Mean']) - 1,
    'SR_green': lambda rr: safe_divide(rr['NIR_Mean'], rr['Green_Mean']),
    # 红边相关指数
    'GRedEdge': lambda rr: safe_divide(rr['Red_Edge_Mean'] - rr['Green_Mean'], rr['Red_Edge_Mean'] + rr['Green_Mean']),
    'NDVI_rededge': lambda rr: safe_divide(rr['NIR_Mean'] - rr['Red_Edge_Mean'], rr['NIR_Mean'] + rr['Red_Edge_Mean']),
    'VIPA': lambda rr: safe_divide(rr['Red_Edge_Mean'] - rr['Red_Mean'], rr['Red_Edge_Mean'] + rr['Red_Mean']),
    'CARI': lambda rr: (rr['Red_Edge_Mean'] - rr['Red_Mean'] - 0.2 * (rr['Red_Edge_Mean'] - rr['Green_Mean'])) *
                       safe_divide(rr['Red_Edge_Mean'], rr['Red_Mean']),
    # 增强型指数
    'SAVI': lambda rr: (1 + 0.5) * safe_divide(rr['NIR_Mean'] - rr['Red_Mean'], rr['NIR_Mean'] + rr['Red_Mean'] + 0.5),
    'MSAVI': lambda rr: safe_divide(
        2 * rr['NIR_Mean'] + 1 - np.sqrt((2 * rr['NIR_Mean'] + 1) ** 2 - 8 * (rr['NIR_Mean'] - rr['Red_Mean'])), 2),
    'OSAVI': lambda rr: safe_divide(rr['NIR_Mean'] - rr['Red_Mean'], rr['NIR_Mean'] + rr['Red_Mean'] + 0.16),
    'CVI2': lambda rr: 2.5 * safe_divide(rr['NIR_Mean'] - rr['Red_Edge_Mean'],
                                         rr['NIR_Mean'] + 6 * rr['Red_Edge_Mean'] - 7.5 * rr['Red_Mean'] + 1),
    # 其他实用指数
    'ARI': lambda rr: safe_divide(rr['Green_Mean'] - rr['Red_Edge_Mean'], rr['Green_Mean'] + rr['Red_Edge_Mean']),
    'PBI': lambda rr: safe_divide(rr['Green_Mean'] - rr['NIR_Mean'], rr['Green_Mean'] + rr['NIR_Mean']),
    'LAI_est': lambda rr: 3.618 * rr['NDVI'] - 0.118 if not np.isnan(rr['NDVI']) else np.nan,
    'RDVI': lambda rr: safe_divide(rr['NIR_Mean'] - rr['Red_Mean'], np.sqrt(rr['NIR_Mean'] + rr['Red_Mean'])),
    'DVI': lambda rr: rr['NIR_Mean'] - rr['Red_Mean'],
}


# --- 核心修改：适配你的Excel列名（Longitude/Latitude） ---
def read_rtk_from_excel(excel_path):
    """
    从你的Excel读取RTK经纬度（列名：Longitude=经度，Latitude=纬度）
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel文件不存在：{excel_path}")

    try:
        # 读取Excel的Longitude（A列）和Latitude（B列）
        df = pd.read_excel(excel_path, sheet_name=0, usecols=['Longitude', 'Latitude'])
    except ValueError as e:
        if "Usecols do not match columns" in str(e):
            raise ValueError("Excel中未找到'Longitude'或'Latitude'列，请检查列名！")
        else:
            raise Exception(f"读取Excel失败：{str(e)}")

    # 数据清洗：删除空值、非数值型数据
    df_clean = df.dropna(subset=['Longitude', 'Latitude'])  # 删除空值行
    # 过滤非数值数据
    df_clean = df_clean[
        pd.to_numeric(df_clean['Longitude'], errors='coerce').notna() &
        pd.to_numeric(df_clean['Latitude'], errors='coerce').notna()
        ]
    # 转换为数值型
    df_clean['Longitude'] = pd.to_numeric(df_clean['Longitude'])
    df_clean['Latitude'] = pd.to_numeric(df_clean['Latitude'])

    # 转换为(经度, 纬度)列表
    rtk_points = list(zip(df_clean['Longitude'].tolist(), df_clean['Latitude'].tolist()))

    # 打印读取结果
    print("📊 Excel数据读取结果：")
    print(f"  - 原始数据总行数：{len(df)}")
    print(f"  - 清洗后有效坐标数：{len(rtk_points)}")
    if len(rtk_points) == 0:
        raise ValueError("Excel中无有效经纬度数据！")

    return rtk_points


# 读取Excel中的RTK点
try:
    rtk_points_wgs84 = read_rtk_from_excel(excel_path)
except Exception as e:
    print(f"\n❌ Excel读取失败：{e}")
    exit(1)

# 打印工具信息
print("\n" + "=" * 70)
print(f"📊 单波段多TIFF像素统计工具（适配你的Excel列名）")
print(f"📁 TIF基础路径: {base_path}")
print(f"📋 Excel文件: {excel_path}")
print(f"💾 输出CSV: {output_csv}")
print(f"📍 Excel有效RTK点数量: {len(rtk_points_wgs84)} 个")
print("=" * 70)


# 坐标转换函数（保持原有逻辑）
def convert_coordinates(src_crs, dst_crs, x, y):
    if src_crs == dst_crs:
        return x, y
    try:
        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return transformer.transform(x, y)
    except Exception as e:
        raise ValueError(f"坐标转换失败: {e}")


def main():
    # 打开所有TIF并验证元数据
    src_list = []
    try:
        for idx, tif_path in enumerate(tiff_files):
            src = rasterio.open(tif_path)
            src_list.append(src)
            if idx == 0:
                base_crs = src.crs
                base_width = src.width
                base_height = src.height
                base_nodata = src.nodata
            else:
                if src.crs != base_crs:
                    raise ValueError(f"第{idx + 1}个TIF坐标系不一致！")
                if src.width != base_width or src.height != base_height:
                    raise ValueError(f"第{idx + 1}个TIF尺寸不一致！")
    except Exception as e:
        print(f"\n❌ 打开TIF失败: {e}")
        for src in src_list:
            src.close()
        return

    try:
        # TIF基本信息
        print(f"\n📋 TIF影像信息：")
        print(f"   坐标系: {base_crs} (EPSG:{base_crs.to_epsg() if base_crs else '未知'})")
        print(f"   尺寸: {base_width}列 × {base_height}行")
        nodata = CUSTOM_NODATA if CUSTOM_NODATA is not None else (base_nodata if base_nodata is not None else 0)
        print(f"   NoData值: {nodata}")

        # 坐标转换准备
        rtk_crs_obj = CRS.from_string(RTK_CRS)
        if base_crs != rtk_crs_obj:
            print(f"\n🔄 将RTK点从{RTK_CRS}转换到TIF坐标系")
        else:
            print(f"\n✅ 坐标系一致，无需转换")

        # 处理RTK点
        results = []
        valid_region_count = 0
        valid_point_count = 0
        half_size = SQUARE_SIZE // 2
        full_image_window = Window(0, 0, base_width, base_height)

        for center_idx, (lon, lat) in enumerate(tqdm(rtk_points_wgs84, desc="处理RTK点"), 1):
            # 坐标转换+行列号计算
            try:
                center_x, center_y = convert_coordinates(rtk_crs_obj, base_crs, lon, lat)
                center_row, center_col = src_list[0].index(center_x, center_y)
            except Exception as e:
                print(f"\n⚠️  点{center_idx}（{lon:.6f}, {lat:.6f}）转换失败，跳过: {str(e)[:50]}")
                continue

            # 判断点是否在TIF范围内
            if not (0 <= center_row < base_height and 0 <= center_col < base_width):
                print(f"\n⚠️  点{center_idx}（{lon:.6f}, {lat:.6f}）在TIF外，跳过")
                continue
            valid_point_count += 1

            # 处理5个区域
            for area_name, area_row_offset, area_col_offset, area_code in area_config:
                region_result = {
                    'Region_ID': f"{center_idx}-{area_code}",
                    'Area_Position': area_name,
                    'Original_Point_ID': center_idx,
                    'Center_Longitude_WGS84': round(lon, 8),
                    'Center_Latitude_WGS84': round(lat, 8),
                    'Center_X_Image_CRS': round(center_x, 4),
                    'Center_Y_Image_CRS': round(center_y, 4),
                    'Center_Row': center_row,
                    'Center_Col': center_col,
                    'Region_Row_Off': (center_row - half_size) + area_row_offset,
                    'Region_Col_Off': (center_col - half_size) + area_col_offset
                }

                try:
                    # 区域窗口计算
                    region_row_off = region_result['Region_Row_Off']
                    region_col_off = region_result['Region_Col_Off']
                    region_window = Window(region_col_off, region_row_off, SQUARE_SIZE, SQUARE_SIZE)
                    valid_window = region_window.intersection(full_image_window)
                    region_result['Window_Size_Valid'] = f"{valid_window.width}x{valid_window.height}"

                    # 跳过无效区域
                    if valid_window.width <= 0 or valid_window.height <= 0:
                        for band in band_names:
                            region_result[f'{band}_Mean'] = np.nan
                            if CALCULATE_STD:
                                region_result[f'{band}_Std'] = np.nan
                            if CALCULATE_VALID_PIXELS:
                                region_result[f'{band}_Valid_Pixels'] = 0
                        for idx_name in vegetation_indices.keys():
                            region_result[idx_name] = np.nan
                        results.append(region_result)
                        continue

                    # 读取波段数据+统计
                    is_region_valid = False
                    for band_idx, (band_name, src) in enumerate(zip(band_names, src_list)):
                        raw_data = src.read(1, window=valid_window, masked=False)
                        float_data = raw_data.astype('float32')
                        float_data[float_data == nodata] = np.nan

                        # 计算统计值
                        mean_val = np.nanmean(float_data)
                        region_result[f'{band_name}_Mean'] = round(float(mean_val), 4) if not np.isnan(
                            mean_val) else np.nan
                        if CALCULATE_STD:
                            std_val = np.nanstd(float_data)
                            region_result[f'{band_name}_Std'] = round(float(std_val), 4) if not np.isnan(
                                std_val) else np.nan
                        if CALCULATE_VALID_PIXELS:
                            valid_pixels = np.count_nonzero(~np.isnan(float_data))
                            region_result[f'{band_name}_Valid_Pixels'] = valid_pixels
                            if valid_pixels > 0:
                                is_region_valid = True

                    # 计算植被指数
                    for idx_name, calc_func in vegetation_indices.items():
                        try:
                            idx_value = calc_func(region_result)
                            region_result[idx_name] = round(float(idx_value), 4) if not np.isnan(idx_value) else np.nan
                        except Exception as e:
                            region_result[idx_name] = np.nan
                            print(f"\n⚠️  区域{center_idx}-{area_code}计算{idx_name}失败: {str(e)[:50]}")

                    if is_region_valid:
                        valid_region_count += 1
                    results.append(region_result)

                except Exception as e:
                    error_msg = str(e)[:100]
                    region_result['Error'] = error_msg
                    print(f"\n❌ 区域{center_idx}-{area_code}处理失败: {error_msg}")
                    for band in band_names:
                        region_result[f'{band}_Mean'] = np.nan
                        if CALCULATE_STD:
                            region_result[f'{band}_Std'] = np.nan
                        if CALCULATE_VALID_PIXELS:
                            region_result[f'{band}_Valid_Pixels'] = 0
                    for idx_name in vegetation_indices.keys():
                        region_result[idx_name] = np.nan
                    results.append(region_result)

        # 统计+保存结果
        print(f"\n" + "=" * 70)
        print(f"📊 处理统计:")
        print(f"   1. Excel读取点总数: {len(rtk_points_wgs84)}")
        print(f"   2. TIF内有效点数量: {valid_point_count}")
        print(f"   3. 有效计算区域数: {valid_region_count}")
        print("=" * 70)

        # 生成CSV字段
        csv_fields = [
            'Region_ID', 'Area_Position', 'Original_Point_ID',
            'Center_Longitude_WGS84', 'Center_Latitude_WGS84',
            'Center_X_Image_CRS', 'Center_Y_Image_CRS',
            'Center_Row', 'Center_Col', 'Region_Row_Off', 'Region_Col_Off', 'Window_Size_Valid'
        ]
        for band in band_names:
            csv_fields.append(f'{band}_Mean')
            if CALCULATE_STD:
                csv_fields.append(f'{band}_Std')
            if CALCULATE_VALID_PIXELS:
                csv_fields.append(f'{band}_Valid_Pixels')
        csv_fields.extend(list(vegetation_indices.keys()))
        csv_fields.append('Error')

        # 保存CSV
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, restval='')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n🎉 结果已保存到: {output_csv}")

        # 预览前5个有效区域
        print(f"\n🔍 前5个有效区域预览:")
        preview_fields = [
            'Region_ID', 'Area_Position', 'Green_Mean', 'Red_Mean', 'Red_Edge_Mean', 'NIR_Mean', 'NDVI'
        ]
        print(f"{' | '.join([f'{f:<15}' for f in preview_fields])}")
        print('-' * (15 * len(preview_fields) + (len(preview_fields) - 1) * 3))
        preview_count = 0
        for res in results:
            if not all(np.isnan(res[f'{b}_Mean']) for b in band_names):
                preview_vals = [str(res[f])[:15].ljust(15) for f in preview_fields]
                print(f"{' | '.join(preview_vals)}")
                preview_count += 1
                if preview_count >= 5:
                    break

    except Exception as e:
        print(f"\n❌ 程序失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for src in src_list:
            src.close()


if __name__ == "__main__":
    # 检查依赖
    try:
        import pyproj
        from tqdm import tqdm
        import pandas as pd
    except ImportError as e:
        missing_lib = str(e).split("'")[1]
        print(f"⚠️  缺少依赖: {missing_lib}")
        print(f"请运行: pip install {missing_lib}")
        exit(1)

    main()