#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "Context.h"
#include "EnergyBalanceModel.h"
#include "PhotosynthesisModel.h"
#include "PlantHydraulicsModel.h"
#include "RadiationModel.h"
#include "SolarPosition.h"
#include "StomatalConductanceModel.h"
#include "Visualizer.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

using namespace std;
using namespace helios;
namespace fs = std::filesystem;

struct MetRecord {
    float air_temperature_c = 26.85f;
    float relative_humidity_percent = 50.0f;
    float wind_speed = 2.0f;
    float air_pressure_pa = 101300.0f;
    float par_input = -1.0f;
};

struct LeafData {
    string plant_name;
    uint leaf_id = 0;
    float area = 0.0f;
    vec3 normal;
    vector<float> par_values;
    vector<float> temperature_values;
    vector<float> rh_leaf_values;
    vector<int> wet_state_values;
};

struct DiseaseDailyMetrics {
    int day_index = 1;
    float LWD_h = 0.0f;
    float TLWD_C = 0.0f;
    float IEH_h = 0.0f;
    float night_RH90_total_h = 0.0f;
    float NHH_h = 0.0f;
    float TWP3_h = 0.0f;
    float LWD_x_TLWD = 0.0f;
    int initial_infection = 0;
};

struct HourlySeriesSummary {
    string name;
    string unit;
    vector<float> mean_values;
    vector<float> min_values;
    vector<float> max_values;
};

static float tetensPa(float temperature_c) {
    return 610.7f * expf(17.38f * temperature_c / (239.0f + temperature_c));
}

static float longwaveRadiationFlux(float temperature_k) {
    const float sigma = 5.670374419e-8f;
    return 2.0f * sigma * powf(temperature_k, 4.0f);
}

static vector<string> splitCSVLine(const string& line) {
    vector<string> values;
    string value;
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            values.push_back(value);
            value.clear();
        } else {
            value.push_back(ch);
        }
    }
    values.push_back(value);
    return values;
}

static bool parseFloat(const string& text, float& out) {
    try {
        size_t pos = 0;
        out = stof(text, &pos);
        return pos > 0;
    } catch (...) {
        return false;
    }
}

static string normalizeHeader(string text) {
    string normalized;
    for (char ch : text) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch)) {
            normalized.push_back(static_cast<char>(std::tolower(uch)));
        }
    }
    return normalized;
}

static int findColumnIndex(const vector<string>& headers, const vector<string>& keys) {
    for (size_t i = 0; i < headers.size(); ++i) {
        string name = normalizeHeader(headers[i]);
        for (const string& key : keys) {
            if (name.find(key) != string::npos) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

static bool parseCellFloat(const vector<string>& cells, int index, float& value) {
    if (index < 0 || index >= static_cast<int>(cells.size())) {
        return false;
    }
    return parseFloat(cells[index], value);
}

static string findNvOptixDll() {
    vector<fs::path> direct_paths = {
            "C:/Windows/System32/nvoptix.dll",
            "C:/Windows/SysWOW64/nvoptix.dll"
    };

    for (const fs::path& path : direct_paths) {
        if (fs::exists(path)) {
            return path.string();
        }
    }

    fs::path driver_store = "C:/Windows/System32/DriverStore/FileRepository";
    if (fs::exists(driver_store)) {
        fs::directory_options options = fs::directory_options::skip_permission_denied;
        for (const auto& entry : fs::recursive_directory_iterator(driver_store, options)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            string filename = normalizeHeader(entry.path().filename().string());
            if (filename == "nvoptixdll") {
                return entry.path().string();
            }
        }
    }

    return "";
}

static string findSystemNvOptixDll() {
    fs::path system_optix = "C:/Windows/System32/nvoptix.dll";
    if (fs::exists(system_optix)) {
        return system_optix.string();
    }
    return "";
}

static void addDllSearchDirectory(const string& dll_path) {
#ifdef _WIN32
    if (dll_path.empty()) {
        return;
    }
    fs::path folder = fs::path(dll_path).parent_path();
    SetDllDirectoryA(folder.string().c_str());
#else
    (void)dll_path;
#endif
}

static vector<MetRecord> readMeteorologyCSV(const string& path) {
    vector<MetRecord> records;
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "WARNING: meteorology input file not found: " << path << endl;
        return records;
    }

    string line;
    vector<string> headers;
    bool header_checked = false;
    bool use_header_columns = false;
    int temp_col = -1;
    int rh_col = -1;
    int wind_col = -1;
    int pressure_col = -1;
    int par_col = -1;

    while (getline(fin, line)) {
        vector<string> cells = splitCSVLine(line);

        if (!header_checked) {
            header_checked = true;
            headers = cells;
            temp_col = findColumnIndex(headers, {"airtemperature", "temperature", "temp", "tair"});
            rh_col = findColumnIndex(headers, {"relativehumidity", "humidity", "rh"});
            wind_col = findColumnIndex(headers, {"windspeed", "wind"});
            pressure_col = findColumnIndex(headers, {"airpressure", "pressure", "press"});
            par_col = findColumnIndex(headers, {"parinput", "par", "paer"});

            use_header_columns = temp_col >= 0 && rh_col >= 0 && wind_col >= 0 && pressure_col >= 0;
            if (use_header_columns) {
                cout << "Meteorology CSV header detected. PAR column index: " << par_col << endl;
                continue;
            }
        }

        if (use_header_columns) {
            MetRecord record;
            float value = 0.0f;
            if (!parseCellFloat(cells, temp_col, record.air_temperature_c)) {
                continue;
            }
            if (!parseCellFloat(cells, rh_col, record.relative_humidity_percent)) {
                continue;
            }
            if (!parseCellFloat(cells, wind_col, record.wind_speed)) {
                continue;
            }
            if (!parseCellFloat(cells, pressure_col, record.air_pressure_pa)) {
                continue;
            }
            if (parseCellFloat(cells, par_col, value)) {
                record.par_input = value;
            }
            records.push_back(record);
            continue;
        }

        vector<float> numeric;
        for (const string& cell : cells) {
            float value = 0.0f;
            if (parseFloat(cell, value)) {
                numeric.push_back(value);
            }
        }

        if (numeric.size() < 4) {
            continue;
        }

        MetRecord record;
        record.air_temperature_c = numeric[0];
        record.relative_humidity_percent = numeric[1];
        record.wind_speed = numeric[2];
        record.air_pressure_pa = numeric[3];
        if (numeric.size() >= 5) {
            record.par_input = numeric[4];
        }
        records.push_back(record);
    }

    return records;
}

static vector<MetRecord> aggregateTo24Hours(const vector<MetRecord>& raw) {
    vector<MetRecord> hourly(24);
    vector<int> counts(24, 0);
    vector<int> par_counts(24, 0);
    for (MetRecord& record : hourly) {
        record.air_temperature_c = 0.0f;
        record.relative_humidity_percent = 0.0f;
        record.wind_speed = 0.0f;
        record.air_pressure_pa = 0.0f;
        record.par_input = 0.0f;
    }

    if (raw.empty()) {
        cerr << "WARNING: no valid meteorology rows were read; using default 24-hour input." << endl;
        vector<MetRecord> defaults(24);
        return defaults;
    }

    for (size_t i = 0; i < raw.size(); ++i) {
        int hour = 0;
        if (raw.size() >= 288) {
            hour = static_cast<int>(i / 12);
        } else {
            hour = static_cast<int>((i * 24) / raw.size());
        }
        hour = std::max(0, std::min(23, hour));

        hourly[hour].air_temperature_c += raw[i].air_temperature_c;
        hourly[hour].relative_humidity_percent += raw[i].relative_humidity_percent;
        hourly[hour].wind_speed += raw[i].wind_speed;
        hourly[hour].air_pressure_pa += raw[i].air_pressure_pa;
        if (raw[i].par_input >= 0.0f) {
            hourly[hour].par_input += raw[i].par_input;
            par_counts[hour]++;
        }
        counts[hour]++;
    }

    for (int h = 0; h < 24; ++h) {
        if (counts[h] == 0) {
            continue;
        }
        hourly[h].air_temperature_c /= counts[h];
        hourly[h].relative_humidity_percent /= counts[h];
        hourly[h].wind_speed /= counts[h];
        hourly[h].air_pressure_pa /= counts[h];
        if (par_counts[h] > 0) {
            hourly[h].par_input /= par_counts[h];
        } else {
            hourly[h].par_input = -1.0f;
        }
    }

    return hourly;
}

static void setLeafParameters(Context& context, const vector<uint>& leaves, const MetRecord& met) {
    const float temperature_k = met.air_temperature_c + 273.15f;
    const float humidity_fraction = std::max(0.0f, std::min(1.0f, met.relative_humidity_percent / 100.0f));
    const float es_air = tetensPa(met.air_temperature_c);
    const float ea = es_air * humidity_fraction;
    const float vpd_kpa = std::max(0.0f, (es_air - ea) / 1000.0f);

    context.setPrimitiveData(leaves, "wind_speed", met.wind_speed);
    context.setPrimitiveData(leaves, "surface_humidity", humidity_fraction);
    context.setPrimitiveData(leaves, "humidity", humidity_fraction);
    context.setPrimitiveData(leaves, "air_humidity", humidity_fraction);
    context.setPrimitiveData(leaves, "air_pressure", met.air_pressure_pa);
    context.setPrimitiveData(leaves, "pressure", met.air_pressure_pa);
    context.setPrimitiveData(leaves, "air_temperature", temperature_k);
    context.setPrimitiveData(leaves, "temperature", temperature_k);
    context.setPrimitiveData(leaves, "vapor_pressure_deficit", vpd_kpa);

    context.setPrimitiveData(leaves, "reflectivity_PAR", 0.30f);
    context.setPrimitiveData(leaves, "transmissivity_PAR", 0.30f);
    context.setPrimitiveData(leaves, "emissivity_PAR", 1.0f);
    context.setPrimitiveData(leaves, "emissivity_LW", 1.0f);
    context.setPrimitiveData(leaves, "radiation_flux_LW", longwaveRadiationFlux(temperature_k));
    context.setPrimitiveData(leaves, "Vcmax25", 60.0f);
    context.setPrimitiveData(leaves, "Jmax25", 120.0f);
    context.setPrimitiveData(leaves, "TPU25", 8.0f);
    context.setPrimitiveData(leaves, "Rd25", 1.5f);
    context.setPrimitiveData(leaves, "theta_J", 0.7f);
    context.setPrimitiveData(leaves, "alpha", 0.24f);
    context.setPrimitiveData(leaves, "g0", 0.01f);
    context.setPrimitiveData(leaves, "g1", 9.0f);
    context.setPrimitiveData(leaves, "D0", 1.5f);
    context.setPrimitiveData(leaves, "air_CO2", 400.0f);
    context.setPrimitiveData(leaves, "beta_soil", 1.0f);
    context.setPrimitiveData(leaves, "Gamma_CO2", 100.0f);
    context.setPrimitiveData(leaves, "leaf_type", string("C3"));
    context.setPrimitiveData(leaves, "stomatal_side", string("amphistomatous"));
    context.setPrimitiveData(leaves, "boundarylayer_conductance", 0.1f);
    context.setPrimitiveData(leaves, "moisture_conductance", 0.25f);
    context.setPrimitiveData(leaves, "twosided_flag", uint(1));
    context.setPrimitiveData(leaves, "stomatal_sidedness", 0.0f);
    context.setPrimitiveData(leaves, "leaf_thickness", 0.0002f);
    context.setPrimitiveData(leaves, "water_potential", -0.5f);
    context.setPrimitiveData(leaves, "relative_water_content", 0.8f);
    context.setPrimitiveData(leaves, "turgor_pressure", 0.3f);
    context.setPrimitiveData(leaves, "osmotic_potential", -0.8f);
    context.setPrimitiveData(leaves, "latent_flux", 0.0f);
}

static void writeTimeseries(
        const string& output_path,
        const string& value_prefix,
        const vector<int>& hours,
        const map<string, vector<LeafData>>& plant_leaf_data,
        const string& type
) {
    ofstream out(output_path);
    if (!out.is_open()) {
        cerr << "WARNING: could not write output file: " << output_path << endl;
        return;
    }

    out << "Plant_Name\tLeaf_ID\tLeaf_Area\tNormal_X\tNormal_Y\tNormal_Z";
    for (int hour : hours) {
        out << "\t" << value_prefix << hour;
    }
    out << '\n';

    for (const auto& plant_pair : plant_leaf_data) {
        for (const LeafData& leaf : plant_pair.second) {
            out << leaf.plant_name << "\t"
                << leaf.leaf_id << "\t"
                << fixed << setprecision(6) << leaf.area << "\t"
                << leaf.normal.x << "\t"
                << leaf.normal.y << "\t"
                << leaf.normal.z;

            for (size_t i = 0; i < hours.size(); ++i) {
                if (type == "PAR") {
                    out << "\t" << leaf.par_values[i];
                } else if (type == "TLEAF") {
                    out << "\t" << leaf.temperature_values[i];
                } else if (type == "RHLEAF") {
                    out << "\t" << leaf.rh_leaf_values[i];
                } else if (type == "WET") {
                    out << "\t" << leaf.wet_state_values[i];
                }
            }
            out << '\n';
        }
    }
}

static bool isNightHour(float hour_of_day) {
    return hour_of_day >= 18.0f || hour_of_day < 6.0f;
}

static vector<DiseaseDailyMetrics> writeDiseaseRiskMetrics(
        const string& base_path,
        const vector<MetRecord>& raw_records
) {
    vector<DiseaseDailyMetrics> daily_metrics;
    if (raw_records.empty()) {
        cerr << "WARNING: no raw meteorology records available for disease risk metrics." << endl;
        return daily_metrics;
    }

    const int records_per_day = 288;
    const float interval_h = 5.0f / 60.0f;
    const int day_count = static_cast<int>((raw_records.size() + records_per_day - 1) / records_per_day);

    vector<float> daily_lwd(day_count, 0.0f);
    vector<float> daily_wet_temp_sum(day_count, 0.0f);
    vector<int> daily_wet_count(day_count, 0);

    ofstream detail_out(base_path + "Disease_5min_Result.txt");
    if (detail_out.is_open()) {
        detail_out << "Day\tIndex_5min\tHour\tTair_C\tRHair_percent\tLW\tLWD_contribution_h\tIEH_flag\tIEH_contribution_h\n";
    }

    for (size_t i = 0; i < raw_records.size(); ++i) {
        int day = static_cast<int>(i / records_per_day);
        int index_in_day = static_cast<int>(i % records_per_day);
        float hour_of_day = static_cast<float>(index_in_day) * interval_h;
        const MetRecord& record = raw_records[i];

        int LW = record.relative_humidity_percent >= 90.0f ? 1 : 0;
        float LWD_contribution_h = LW ? interval_h : 0.0f;

        int IEH_flag = (record.air_temperature_c >= 15.0f &&
                        record.air_temperature_c <= 22.0f &&
                        record.relative_humidity_percent >= 85.0f) ? 1 : 0;
        float IEH_contribution_h = IEH_flag ? interval_h : 0.0f;

        daily_lwd[day] += LWD_contribution_h;
        if (LW) {
            daily_wet_temp_sum[day] += record.air_temperature_c;
            daily_wet_count[day]++;
        }

        if (detail_out.is_open()) {
            detail_out << (day + 1) << '\t'
                       << index_in_day << '\t'
                       << fixed << setprecision(4) << hour_of_day << '\t'
                       << record.air_temperature_c << '\t'
                       << record.relative_humidity_percent << '\t'
                       << LW << '\t'
                       << LWD_contribution_h << '\t'
                       << IEH_flag << '\t'
                       << IEH_contribution_h << '\n';
        }
    }

    daily_metrics.resize(day_count);
    for (int day = 0; day < day_count; ++day) {
        DiseaseDailyMetrics metrics;
        metrics.day_index = day + 1;
        metrics.LWD_h = daily_lwd[day];
        metrics.TLWD_C = daily_wet_count[day] > 0 ? daily_wet_temp_sum[day] / daily_wet_count[day] : 0.0f;

        float current_night_rh90_h = 0.0f;
        float max_night_rh90_h = 0.0f;
        int begin = day * records_per_day;
        int end = std::min(static_cast<int>(raw_records.size()), begin + records_per_day);

        for (int i = begin; i < end; ++i) {
            int index_in_day = i - begin;
            float hour_of_day = static_cast<float>(index_in_day) * interval_h;
            const MetRecord& record = raw_records[i];

            bool IEH = record.air_temperature_c >= 15.0f &&
                       record.air_temperature_c <= 22.0f &&
                       record.relative_humidity_percent >= 85.0f;
            if (IEH) {
                metrics.IEH_h += interval_h;
            }

            bool night_rh90 = isNightHour(hour_of_day) && record.relative_humidity_percent >= 90.0f;
            if (night_rh90) {
                metrics.night_RH90_total_h += interval_h;
                current_night_rh90_h += interval_h;
                max_night_rh90_h = std::max(max_night_rh90_h, current_night_rh90_h);
            } else if (isNightHour(hour_of_day)) {
                current_night_rh90_h = 0.0f;
            }
        }

        metrics.NHH_h = max_night_rh90_h;
        int start_day = std::max(0, day - 2);
        for (int d = start_day; d <= day; ++d) {
            metrics.TWP3_h += daily_lwd[d];
        }
        metrics.LWD_x_TLWD = metrics.LWD_h * metrics.TLWD_C;
        metrics.initial_infection = metrics.LWD_x_TLWD >= 40.0f ? 1 : 0;
        daily_metrics[day] = metrics;
    }

    ofstream daily_out(base_path + "Disease_Daily_Result.txt");
    if (daily_out.is_open()) {
        daily_out << "Day\tLWD_h\tTLWD_C\tIEH_h\tNight_RH90_total_h\tNHH_h\tTWP3_h\tLWD_x_TLWD\tInitialInfection\n";
        for (const DiseaseDailyMetrics& metrics : daily_metrics) {
            daily_out << metrics.day_index << '\t'
                      << fixed << setprecision(4)
                      << metrics.LWD_h << '\t'
                      << metrics.TLWD_C << '\t'
                      << metrics.IEH_h << '\t'
                      << metrics.night_RH90_total_h << '\t'
                      << metrics.NHH_h << '\t'
                      << metrics.TWP3_h << '\t'
                      << metrics.LWD_x_TLWD << '\t'
                      << metrics.initial_infection << '\n';
        }
    } else {
        cerr << "WARNING: could not write disease daily metrics file." << endl;
    }

    return daily_metrics;
}

static string escapeJSString(const string& text) {
    string escaped;
    for (char ch : text) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

static string quotedStringArray(const vector<string>& values) {
    ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "\"" << escapeJSString(values[i]) << "\"";
    }
    out << "]";
    return out.str();
}

static string floatArrayLiteral(const vector<float>& values, int precision = 4) {
    ostringstream out;
    out << "[";
    out << fixed << setprecision(precision);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

static string intArrayLiteral(const vector<int>& values) {
    ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

static vector<string> buildHourLabels(const vector<int>& hours) {
    vector<string> labels;
    labels.reserve(hours.size());
    for (int hour : hours) {
        labels.push_back("Hour " + to_string(hour));
    }
    return labels;
}

static HourlySeriesSummary summarizeLeafMetric(
        const map<string, vector<LeafData>>& plant_leaf_data,
        const vector<int>& hours,
        const string& metric_type,
        const string& display_name,
        const string& unit
) {
    HourlySeriesSummary summary;
    summary.name = display_name;
    summary.unit = unit;
    summary.mean_values.resize(hours.size(), 0.0f);
    summary.min_values.resize(hours.size(), 0.0f);
    summary.max_values.resize(hours.size(), 0.0f);

    for (size_t h = 0; h < hours.size(); ++h) {
        bool initialized = false;
        float sum = 0.0f;
        int count = 0;
        float min_value = 0.0f;
        float max_value = 0.0f;

        for (const auto& plant_pair : plant_leaf_data) {
            for (const LeafData& leaf : plant_pair.second) {
                float value = 0.0f;
                if (metric_type == "PAR") {
                    value = leaf.par_values[h];
                } else if (metric_type == "TLEAF") {
                    value = leaf.temperature_values[h];
                } else if (metric_type == "RHLEAF") {
                    value = leaf.rh_leaf_values[h];
                } else if (metric_type == "WET") {
                    value = static_cast<float>(leaf.wet_state_values[h]);
                } else {
                    continue;
                }

                if (!initialized) {
                    min_value = value;
                    max_value = value;
                    initialized = true;
                } else {
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
                sum += value;
                count++;
            }
        }

        if (count > 0) {
            summary.mean_values[h] = sum / static_cast<float>(count);
            summary.min_values[h] = min_value;
            summary.max_values[h] = max_value;
        }
    }

    return summary;
}

static size_t indexOfMaxValue(const vector<float>& values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

static void applySceneMetricSnapshot(
        Context& context,
        const map<string, vector<uint>>& plant_prims,
        const map<string, vector<LeafData>>& plant_leaf_data,
        const vector<uint>& greenhouse_wall,
        const vector<uint>& greenhouse_roof,
        const vector<uint>& greenhouse_floor,
        const vector<MetRecord>& hourly_met,
        size_t hour_index,
        const string& metric_type,
        const string& target_data_name
) {
    hour_index = std::min(hour_index, hourly_met.empty() ? size_t(0) : hourly_met.size() - 1);
    const MetRecord& met = hourly_met[hour_index];

    float structure_value = 0.0f;
    if (metric_type == "PAR") {
        structure_value = std::max(0.0f, met.par_input);
    } else if (metric_type == "TLEAF") {
        structure_value = met.air_temperature_c;
    } else if (metric_type == "RHLEAF") {
        structure_value = met.relative_humidity_percent;
    } else if (metric_type == "WET") {
        structure_value = met.relative_humidity_percent >= 95.0f ? 1.0f : 0.0f;
    }

    for (const auto& plant_pair : plant_prims) {
        const vector<uint>& leaves = plant_pair.second;
        const vector<LeafData>& leaf_data = plant_leaf_data.at(plant_pair.first);
        for (size_t i = 0; i < leaves.size() && i < leaf_data.size(); ++i) {
            float value = 0.0f;
            if (metric_type == "PAR") {
                value = leaf_data[i].par_values[hour_index];
            } else if (metric_type == "TLEAF") {
                value = leaf_data[i].temperature_values[hour_index];
            } else if (metric_type == "RHLEAF") {
                value = leaf_data[i].rh_leaf_values[hour_index];
            } else if (metric_type == "WET") {
                value = static_cast<float>(leaf_data[i].wet_state_values[hour_index]);
            }
            context.setPrimitiveData(leaves[i], target_data_name.c_str(), value);
        }
    }

    if (!greenhouse_wall.empty()) {
        context.setPrimitiveData(greenhouse_wall, target_data_name.c_str(), structure_value);
    }
    if (!greenhouse_roof.empty()) {
        context.setPrimitiveData(greenhouse_roof, target_data_name.c_str(), structure_value);
    }
    if (!greenhouse_floor.empty()) {
        context.setPrimitiveData(greenhouse_floor, target_data_name.c_str(), structure_value);
    }
}

static void writeVisualizationPage(
        const string& output_path,
        const string& page_title,
        const vector<string>& labels,
        const HourlySeriesSummary& summary,
        const string& chart_type,
        const string& color_hex,
        const string& description
) {
    ofstream out(output_path);
    if (!out.is_open()) {
        cerr << "WARNING: could not write visualization page: " << output_path << endl;
        return;
    }

    out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        << "<meta charset=\"UTF-8\">\n"
        << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        << "<title>" << page_title << "</title>\n"
        << "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n"
        << "<style>\n"
        << "body{font-family:'Times New Roman',Times,serif;margin:0;background:#f5f7fb;color:#1f2937;}\n"
        << ".wrap{max-width:1200px;margin:0 auto;padding:24px;}\n"
        << ".panel{background:#fff;border-radius:12px;padding:20px;box-shadow:0 8px 24px rgba(15,23,42,.08);margin-bottom:20px;}\n"
        << ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;}\n"
        << ".card{background:#eef4ff;border-radius:10px;padding:16px;}\n"
        << ".label{font-size:13px;color:#475569;margin-bottom:8px;}\n"
        << ".value{font-size:28px;font-weight:700;color:#0f172a;}\n"
        << "canvas{width:100%!important;height:440px!important;}\n"
        << "@media (max-width:900px){.grid{grid-template-columns:1fr;}}\n"
        << "</style>\n</head>\n<body>\n<div class=\"wrap\">\n";

    float overall_min = 0.0f;
    float overall_max = 0.0f;
    float overall_mean = 0.0f;
    int count = 0;
    bool initialized = false;
    for (float value : summary.mean_values) {
        if (!initialized) {
            overall_min = value;
            overall_max = value;
            initialized = true;
        } else {
            overall_min = std::min(overall_min, value);
            overall_max = std::max(overall_max, value);
        }
        overall_mean += value;
        count++;
    }
    if (count > 0) {
        overall_mean /= static_cast<float>(count);
    }

    out << "<div class=\"panel\"><h1>" << page_title << "</h1><p>" << description << "</p></div>\n";
    out << "<div class=\"grid\">\n";
    out << "<div class=\"card\"><div class=\"label\">Average mean</div><div class=\"value\">" << fixed << setprecision(2)
        << overall_mean << " " << summary.unit << "</div></div>\n";
    out << "<div class=\"card\"><div class=\"label\">Lowest hourly mean</div><div class=\"value\">" << overall_min
        << " " << summary.unit << "</div></div>\n";
    out << "<div class=\"card\"><div class=\"label\">Highest hourly mean</div><div class=\"value\">" << overall_max
        << " " << summary.unit << "</div></div>\n";
    out << "</div>\n";

    out << "<div class=\"panel\"><canvas id=\"metricChart\"></canvas></div>\n";
    out << "</div>\n<script>\n";
    out << "const labels=" << quotedStringArray(labels) << ";\n";
    out << "const meanValues=" << floatArrayLiteral(summary.mean_values) << ";\n";
    out << "const minValues=" << floatArrayLiteral(summary.min_values) << ";\n";
    out << "const maxValues=" << floatArrayLiteral(summary.max_values) << ";\n";
    out << "new Chart(document.getElementById('metricChart'),{type:'" << chart_type << "',data:{labels:labels,datasets:["
        << "{label:'Mean',data:meanValues,borderColor:'" << color_hex << "',backgroundColor:'rgba(59,130,246,0.15)',borderWidth:3,tension:0.28,fill:false},"
        << "{label:'Min',data:minValues,borderColor:'rgba(16,185,129,0.85)',backgroundColor:'rgba(16,185,129,0.12)',borderWidth:2,tension:0.22,fill:false},"
        << "{label:'Max',data:maxValues,borderColor:'rgba(239,68,68,0.85)',backgroundColor:'rgba(239,68,68,0.12)',borderWidth:2,tension:0.22,fill:false}"
        << "]},options:{responsive:true,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'top'}},scales:{y:{title:{display:true,text:'"
        << escapeJSString(summary.name + " (" + summary.unit + ")") << "'}}}}});\n";
    out << "</script>\n</body>\n</html>\n";
}

static void writeDiseaseVisualizationPage(
        const string& output_path,
        const vector<DiseaseDailyMetrics>& disease_metrics
) {
    ofstream out(output_path);
    if (!out.is_open()) {
        cerr << "WARNING: could not write disease visualization page: " << output_path << endl;
        return;
    }

    vector<string> labels;
    vector<float> lwd_values;
    vector<float> ieh_values;
    vector<float> nlh_values;
    vector<float> risk_values;
    vector<int> infection_flags;
    for (const DiseaseDailyMetrics& metrics : disease_metrics) {
        labels.push_back("Day " + to_string(metrics.day_index));
        lwd_values.push_back(metrics.LWD_h);
        ieh_values.push_back(metrics.IEH_h);
        nlh_values.push_back(metrics.night_RH90_total_h);
        risk_values.push_back(metrics.LWD_x_TLWD);
        infection_flags.push_back(metrics.initial_infection);
    }

    out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        << "<meta charset=\"UTF-8\">\n"
        << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        << "<title>Disease Risk Metrics</title>\n"
        << "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n"
        << "<style>\n"
        << "body{font-family:'Times New Roman',Times,serif;margin:0;background:#f5f7fb;color:#1f2937;}\n"
        << ".wrap{max-width:1200px;margin:0 auto;padding:24px;}\n"
        << ".panel{background:#fff;border-radius:12px;padding:20px;box-shadow:0 8px 24px rgba(15,23,42,.08);margin-bottom:20px;}\n"
        << "canvas{width:100%!important;height:420px!important;}\n"
        << "</style>\n</head>\n<body>\n<div class=\"wrap\">\n";
    out << "<div class=\"panel\"><h1>Disease Risk Metrics</h1><p>Daily disease risk indicators generated from meteorology and leaf wetness conditions.</p></div>\n";
    out << "<div class=\"panel\"><canvas id=\"diseaseChart\"></canvas></div>\n";
    out << "<div class=\"panel\"><canvas id=\"infectionChart\"></canvas></div>\n";
    out << "</div>\n<script>\n";
    out << "const labels=" << quotedStringArray(labels) << ";\n";
    out << "new Chart(document.getElementById('diseaseChart'),{type:'bar',data:{labels:labels,datasets:["
        << "{label:'LWD_h',data:" << floatArrayLiteral(lwd_values) << ",backgroundColor:'rgba(59,130,246,0.75)'},"
        << "{label:'IEH_h',data:" << floatArrayLiteral(ieh_values) << ",backgroundColor:'rgba(16,185,129,0.75)'},"
        << "{label:'Night_RH90_h',data:" << floatArrayLiteral(nlh_values) << ",backgroundColor:'rgba(245,158,11,0.75)'},"
        << "{label:'LWD_x_TLWD',data:" << floatArrayLiteral(risk_values) << ",backgroundColor:'rgba(239,68,68,0.75)'}"
        << "]},options:{responsive:true,plugins:{legend:{position:'top'}},scales:{y:{beginAtZero:true}}}});\n";
    out << "new Chart(document.getElementById('infectionChart'),{type:'line',data:{labels:labels,datasets:["
        << "{label:'Initial infection flag',data:" << intArrayLiteral(infection_flags) << ",borderColor:'#7c3aed',backgroundColor:'rgba(124,58,237,0.2)',borderWidth:3,tension:0.15,fill:true}"
        << "]},options:{responsive:true,plugins:{legend:{position:'top'}},scales:{y:{min:0,max:1,title:{display:true,text:'0 = no, 1 = yes'}}}}});\n";
    out << "</script>\n</body>\n</html>\n";
}

static void writeMeteorologyVisualizationPage(
        const string& output_path,
        const vector<MetRecord>& hourly_met
) {
    ofstream out(output_path);
    if (!out.is_open()) {
        cerr << "WARNING: could not write meteorology visualization page: " << output_path << endl;
        return;
    }

    vector<string> labels;
    vector<float> tair_values;
    vector<float> rh_values;
    vector<float> par_values;
    for (size_t i = 0; i < hourly_met.size(); ++i) {
        labels.push_back("Hour " + to_string(i));
        tair_values.push_back(hourly_met[i].air_temperature_c);
        rh_values.push_back(hourly_met[i].relative_humidity_percent);
        par_values.push_back(hourly_met[i].par_input);
    }

    out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        << "<meta charset=\"UTF-8\">\n"
        << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        << "<title>Meteorology Overview</title>\n"
        << "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n"
        << "<style>\n"
        << "body{font-family:'Times New Roman',Times,serif;margin:0;background:#f5f7fb;color:#1f2937;}\n"
        << ".wrap{max-width:1200px;margin:0 auto;padding:24px;}\n"
        << ".panel{background:#fff;border-radius:12px;padding:20px;box-shadow:0 8px 24px rgba(15,23,42,.08);margin-bottom:20px;}\n"
        << "canvas{width:100%!important;height:420px!important;}\n"
        << "</style>\n</head>\n<body>\n<div class=\"wrap\">\n";
    out << "<div class=\"panel\"><h1>Meteorology Overview</h1><p>Hourly air temperature, relative humidity and input PAR used by the simulation.</p></div>\n";
    out << "<div class=\"panel\"><canvas id=\"metChart\"></canvas></div>\n";
    out << "</div>\n<script>\n";
    out << "const labels=" << quotedStringArray(labels) << ";\n";
    out << "new Chart(document.getElementById('metChart'),{type:'line',data:{labels:labels,datasets:["
        << "{label:'Air temperature (C)',data:" << floatArrayLiteral(tair_values) << ",borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.15)',borderWidth:3,tension:0.25,yAxisID:'y'},"
        << "{label:'Relative humidity (%)',data:" << floatArrayLiteral(rh_values) << ",borderColor:'#0ea5e9',backgroundColor:'rgba(14,165,233,0.15)',borderWidth:3,tension:0.25,yAxisID:'y1'},"
        << "{label:'PAR input',data:" << floatArrayLiteral(par_values) << ",borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,0.15)',borderWidth:3,tension:0.25,yAxisID:'y2'}"
        << "]},options:{responsive:true,interaction:{mode:'index',intersect:false},scales:{y:{type:'linear',position:'left'},y1:{type:'linear',position:'right',grid:{drawOnChartArea:false}},y2:{type:'linear',position:'right',grid:{drawOnChartArea:false}}}}});\n";
    out << "</script>\n</body>\n</html>\n";
}

static void writeVisualizationIndexPage(
        const string& output_path,
        const vector<pair<string, string>>& pages
) {
    ofstream out(output_path);
    if (!out.is_open()) {
        cerr << "WARNING: could not write visualization index page: " << output_path << endl;
        return;
    }

    out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        << "<meta charset=\"UTF-8\">\n"
        << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        << "<title>Simulation Visualization Index</title>\n"
        << "<style>\n"
        << "body{font-family:'Times New Roman',Times,serif;margin:0;background:#f5f7fb;color:#1f2937;}\n"
        << ".wrap{max-width:1100px;margin:0 auto;padding:24px;}\n"
        << ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;}\n"
        << ".card{display:block;background:#fff;border-radius:12px;padding:20px;text-decoration:none;color:#0f172a;box-shadow:0 8px 24px rgba(15,23,42,.08);}\n"
        << ".card h2{margin:0 0 10px 0;font-size:20px;}\n"
        << ".card p{margin:0;color:#475569;}\n"
        << "@media (max-width:900px){.grid{grid-template-columns:1fr;}}\n"
        << "</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
        << "<h1>Simulation Visualization Pages</h1>\n"
        << "<div class=\"grid\">\n";
    for (const auto& page : pages) {
        out << "<a class=\"card\" href=\"" << page.second << "\"><h2>" << page.first
            << "</h2><p>Open visualization page</p></a>\n";
    }
    out << "</div>\n</div>\n</body>\n</html>\n";
}

static void showVisualization(Context& context, const string& data_name, float min_value, float max_value) {
    try {
        Visualizer vis(800);
        RGBcolor background = RGBcolor(0, 0, 0);
        RGBcolor text_color = RGBcolor(255, 255, 255);

        vis.setBackgroundColor(background);
        vis.setColormap(Visualizer::COLORMAP_RAINBOW);
        vis.setColorbarRange(min_value, max_value);
        vis.setColorbarSize(make_vec2(0.3f, 0.15f));
        vis.setColorbarPosition(make_vec3(0.5f, 0.15f, 0.0f));
        vis.setColorbarTitle(data_name.c_str());
        vis.setColorbarFontSize(14);
        vis.setColorbarFontColor(text_color);
        vis.buildContextGeometry(&context);
        vis.colorContextPrimitivesByData(data_name.c_str());
        vis.plotInteractive();
    } catch (const std::exception& e) {
        cout << "Visualization failed: " << e.what() << endl;
    }
}

static pair<float, float> getPrimitiveDataRange(
        Context& context,
        const vector<uint>& prims,
        const string& data_name
) {
    bool initialized = false;
    float min_value = 0.0f;
    float max_value = 0.0f;

    for (uint uuid : prims) {
        if (!context.doesPrimitiveDataExist(uuid, data_name.c_str())) {
            continue;
        }
        float value = 0.0f;
        context.getPrimitiveData(uuid, data_name.c_str(), value);
        if (!initialized) {
            min_value = value;
            max_value = value;
            initialized = true;
        } else {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }

    if (!initialized) {
        return {0.0f, 1.0f};
    }
    if (fabs(max_value - min_value) < 1e-6f) {
        max_value = min_value + 1.0f;
    }
    return {min_value, max_value};
}

static pair<float, float> adjustColorbarRange(
        const string& data_name,
        float min_value,
        float max_value
);

static pair<float, float> getSceneColorbarRange(
        Context& context,
        const vector<uint>& leaf_prims,
        const vector<uint>& scene_prims,
        const string& data_name
) {
    pair<float, float> leaf_range = getPrimitiveDataRange(context, leaf_prims, data_name);
    pair<float, float> scene_range = getPrimitiveDataRange(context, scene_prims, data_name);
    pair<float, float> adjusted = adjustColorbarRange(data_name, leaf_range.first, leaf_range.second);

    string key = normalizeHeader(data_name);
    if (key.find("wetstate") != string::npos || key.find("wet_state") != string::npos || key.find("wet") != string::npos) {
        return adjusted;
    }

    if (scene_range.second > adjusted.second) {
        adjusted.second = scene_range.second;
    }
    if (scene_range.first < adjusted.first && (key.find("temperature") == string::npos && key.find("temp") == string::npos)) {
        adjusted.first = std::max(0.0f, scene_range.first);
    }
    return adjusted;
}

static vector<float> buildLinearTicks(float min_value, float max_value, int tick_count) {
    vector<float> ticks;
    if (tick_count < 2) {
        ticks.push_back(min_value);
        return ticks;
    }
    ticks.reserve(tick_count);
    for (int i = 0; i < tick_count; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(tick_count - 1);
        ticks.push_back(min_value + (max_value - min_value) * t);
    }
    return ticks;
}

static string buildColorbarTitleWithRange(
        const string& colorbar_title,
        float min_value,
        float max_value
) {
    ostringstream out;
    out << colorbar_title << "\nRange: " << fixed << setprecision(2) << min_value << " to " << max_value;
    return out.str();
}

static pair<float, float> adjustColorbarRange(
        const string& data_name,
        float min_value,
        float max_value
) {
    string key = normalizeHeader(data_name);
    if (max_value < min_value) {
        std::swap(min_value, max_value);
    }

    float span = max_value - min_value;
    if (span < 1e-6f) {
        span = 1.0f;
        max_value = min_value + span;
    }

    if (key.find("par") != string::npos || key.find("radiation") != string::npos) {
        float pad = std::max(1.0f, span * 0.05f);
        min_value = std::max(0.0f, min_value - pad);
        max_value += pad;
    } else if (key.find("temperature") != string::npos || key.find("temp") != string::npos) {
        float pad = std::max(0.5f, span * 0.10f);
        min_value = std::max(0.0f, min_value - pad);
        max_value = std::min(50.0f, max_value + pad);
        if (max_value - min_value < 2.0f) {
            float center = 0.5f * (min_value + max_value);
            min_value = std::max(0.0f, center - 1.0f);
            max_value = std::min(50.0f, center + 1.0f);
        }
    } else if (key.find("rh") != string::npos || key.find("humidity") != string::npos) {
        min_value = std::clamp(min_value, 0.0f, 100.0f);
        max_value = std::clamp(max_value, 0.0f, 100.0f);
        if (max_value < min_value) {
            max_value = min_value;
        }
        if (max_value - min_value < 5.0f) {
            float center = 0.5f * (min_value + max_value);
            min_value = std::max(0.0f, center - 2.5f);
            max_value = std::min(100.0f, center + 2.5f);
        }
    } else if (key.find("wetstate") != string::npos || key.find("wet_state") != string::npos || key.find("wet") != string::npos) {
        min_value = 0.0f;
        max_value = 2.0f;
    }

    if (max_value - min_value < 1e-6f) {
        max_value = min_value + 1.0f;
    }
    return {min_value, max_value};
}

static void exportSceneVisualizationJPEG(
        Context& context,
        const vector<uint>& leaf_prims,
        const vector<uint>& scene_prims,
        const string& data_name,
        const string& colorbar_title,
        const string& output_path,
        Visualizer::Ctable colormap,
        const helios::vec3& look_at,
        float camera_radius
) {
    pair<float, float> range = getSceneColorbarRange(context, leaf_prims, scene_prims, data_name);
    vector<float> ticks = buildLinearTicks(range.first, range.second, 4);
    string titled_range = buildColorbarTitleWithRange(colorbar_title, range.first, range.second);

    try {
        Visualizer vis(1280);
        vis.hideWatermark();
        vis.setBackgroundColor(RGBcolor(0, 0, 0));
        vis.setLightingModel(Visualizer::LIGHTING_PHONG);
        vis.setLightDirection(sphere2cart(make_SphericalCoord(1.0f, 35.0f * float(M_PI) / 180.0f, 210.0f * float(M_PI) / 180.0f)));
        vis.setCameraFieldOfView(20.0f);
        vis.setCameraPosition(
                make_SphericalCoord(camera_radius, 12.0f * float(M_PI) / 180.0f, 178.0f * float(M_PI) / 180.0f),
                look_at
        );
        vis.enableColorbar();
        vis.setColorbarPosition(make_vec3(0.94f, 0.47f, 0.0f));
        vis.setColorbarSize(make_vec2(0.055f, 0.68f));
        vis.setColorbarRange(range.first, range.second);
        vis.setColorbarTicks(ticks);
        vis.setColorbarTitle(titled_range.c_str());
        vis.setColorbarFontSize(14);
        vis.setColorbarFontColor(RGBcolor(255, 255, 255));
        vis.setColormap(colormap);
        vis.buildContextGeometry(&context, scene_prims);
        vis.colorContextPrimitivesByData(data_name.c_str(), scene_prims);
        vis.plotUpdate(true);
        vis.printWindow(output_path.c_str());
        vis.closeWindow();
    } catch (const std::exception& e) {
        cerr << "WARNING: failed to export scene visualization " << output_path << ": " << e.what() << endl;
    }
}

static void showSceneVisualizationInteractive(
        Context& context,
        const vector<uint>& leaf_prims,
        const vector<uint>& scene_prims,
        const string& data_name,
        const string& colorbar_title,
        Visualizer::Ctable colormap,
        const helios::vec3& look_at,
        float camera_radius
) {
    pair<float, float> range = getSceneColorbarRange(context, leaf_prims, scene_prims, data_name);
    vector<float> ticks = buildLinearTicks(range.first, range.second, 4);
    string titled_range = buildColorbarTitleWithRange(colorbar_title, range.first, range.second);

    try {
        Visualizer vis(1280);
        vis.hideWatermark();
        vis.setBackgroundColor(RGBcolor(0, 0, 0));
        vis.setLightingModel(Visualizer::LIGHTING_PHONG);
        vis.setLightDirection(sphere2cart(make_SphericalCoord(1.0f, 35.0f * float(M_PI) / 180.0f, 210.0f * float(M_PI) / 180.0f)));
        vis.setCameraFieldOfView(20.0f);
        vis.setCameraPosition(
                make_SphericalCoord(camera_radius, 12.0f * float(M_PI) / 180.0f, 178.0f * float(M_PI) / 180.0f),
                look_at
        );
        vis.enableColorbar();
        vis.setColorbarPosition(make_vec3(0.94f, 0.47f, 0.0f));
        vis.setColorbarSize(make_vec2(0.055f, 0.68f));
        vis.setColorbarRange(range.first, range.second);
        vis.setColorbarTicks(ticks);
        vis.setColorbarTitle(titled_range.c_str());
        vis.setColorbarFontSize(14);
        vis.setColorbarFontColor(RGBcolor(255, 255, 255));
        vis.setColormap(colormap);
        vis.buildContextGeometry(&context, scene_prims);
        vis.colorContextPrimitivesByData(data_name.c_str(), scene_prims);
        cout << "Opening interactive scene visualization: " << titled_range << endl;
        cout << "Close this window to continue to the next variable." << endl;
        vis.plotInteractive();
        vis.closeWindow();
    } catch (const std::exception& e) {
        cerr << "WARNING: failed to open interactive scene visualization " << colorbar_title << ": " << e.what() << endl;
    }
}

static vector<uint> loadStructurePLY(
        Context& context,
        const string& path,
        const string& label,
        float reflectivity_par,
        float transmissivity_par,
        float emissivity_lw
) {
    vector<uint> prims;
    if (!fs::exists(path)) {
        cout << label << " model not found: " << path << endl;
        return prims;
    }

    prims = context.loadPLY(path.c_str(), make_vec3(0, 0, 0), 0, "ZUP");
    if (prims.empty()) {
        cout << "WARNING: " << label << " model loaded zero primitives: " << path << endl;
        return prims;
    }

    context.setPrimitiveData(prims, "reflectivity_PAR", reflectivity_par);
    context.setPrimitiveData(prims, "transmissivity_PAR", transmissivity_par);
    context.setPrimitiveData(prims, "emissivity_PAR", 1.0f - reflectivity_par - transmissivity_par);
    context.setPrimitiveData(prims, "emissivity_LW", emissivity_lw);

    cout << label << " model loaded: " << path << endl;
    cout << "  primitives = " << prims.size() << endl;
    cout << "  reflectivity_PAR = " << reflectivity_par << endl;
    cout << "  transmissivity_PAR = " << transmissivity_par << endl;
    cout << "  emissivity_LW = " << emissivity_lw << endl;
    return prims;
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);

    string csv_path = "C:/Users/Che/Desktop/Third paper/test_meteorology.csv";
    string ply_dir = "E:/ymh/document/date/plant";

    string greenhouse_wall_ply = "E:/ymh/document/date/greenhouse_wall.ply";
    string greenhouse_roof_ply = "";

    string base_path = "E:/ymh/document/date/plant/";

    vector<string> positional_args;
    bool optix_required = true;
    bool visualization_enabled = true;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--use-optix") {
            optix_required = true;
        } else if (arg == "--no-optix") {
            optix_required = false;
        } else if (arg == "--vis") {
            visualization_enabled = true;
        } else if (arg == "--no-vis") {
            visualization_enabled = false;
        } else {
            positional_args.push_back(arg);
        }
    }

    if (positional_args.size() > 0) {
        csv_path = positional_args[0];
    }
    if (positional_args.size() > 1) {
        ply_dir = positional_args[1];
        if (!ply_dir.empty() && ply_dir.back() != '/' && ply_dir.back() != '\\') {
            ply_dir += "/";
        }
        base_path = ply_dir;
    }
    if (positional_args.size() > 2) {
        greenhouse_wall_ply = positional_args[2];
    }
    if (positional_args.size() > 3) {
        greenhouse_roof_ply = positional_args[3];
    }

    vector<MetRecord> raw_met = readMeteorologyCSV(csv_path);
    vector<MetRecord> hourly_met = aggregateTo24Hours(raw_met);
    cout << "Meteorology rows read: " << raw_met.size() << endl;
    vector<DiseaseDailyMetrics> disease_metrics = writeDiseaseRiskMetrics(base_path, raw_met);
    cout << "Disease risk metrics generated for " << disease_metrics.size() << " day(s)." << endl;

    Context context;

    // Greenhouse structures are commented out – no wall, roof, or floor will be loaded.
    // Declare empty vectors so that any checks remain safe.
    vector<uint> greenhouse_wall;
    vector<uint> greenhouse_roof;
    vector<uint> greenhouse_floor;

    greenhouse_wall = loadStructurePLY(
            context,
            greenhouse_wall_ply,
            "Greenhouse wall",
            0.35f,
            0.05f,
            0.90f
    );

    map<string, vector<uint>> plant_prims;
    map<string, vector<LeafData>> plant_leaf_data;

    for (const auto& entry : fs::directory_iterator(ply_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".ply" && entry.path().extension() != ".PLY") {
            continue;
        }

        string path = entry.path().string();
        string name = entry.path().filename().string();
        vector<uint> leaves = context.loadPLY(path.c_str(), make_vec3(0, 0, 0), 0, "ZUP");
        if (leaves.empty()) {
            cerr << "WARNING: no primitives loaded from " << name << endl;
            continue;
        }

        setLeafParameters(context, leaves, hourly_met[0]);
        plant_prims[name] = leaves;

        vector<LeafData> rows;
        for (uint leaf_id : leaves) {
            LeafData data;
            data.plant_name = name;
            data.leaf_id = leaf_id;
            data.area = context.getPrimitiveArea(leaf_id);
            data.normal = context.getPrimitiveNormal(leaf_id);
            data.par_values.resize(24, 0.0f);
            data.temperature_values.resize(24, 0.0f);
            data.rh_leaf_values.resize(24, 0.0f);
            data.wet_state_values.resize(24, 0);
            rows.push_back(data);
        }
        plant_leaf_data[name] = rows;
        cout << "Loaded plant: " << name << " primitives: " << leaves.size() << endl;
    }

    vector<uint> all_leaves;
    for (const auto& plant : plant_prims) {
        all_leaves.insert(all_leaves.end(), plant.second.begin(), plant.second.end());
    }
    if (all_leaves.empty()) {
        cerr << "ERROR: no plant PLY primitives were loaded from " << ply_dir << endl;
        return 1;
    }

    vector<int> time_hours;
    for (int h = 0; h < 24; ++h) {
        time_hours.push_back(h);
    }

    Date date(5, 6, 2026);
    context.setDate(date);

    SolarPosition solarposition(8, 39.26, 115.25, &context);
    RadiationModel* radiationmodel = nullptr;
    uint sun_source = 0;
    bool radiation_available = false;
    string optix_dll = findSystemNvOptixDll();
    if (optix_required && optix_dll.empty()) {
        string driver_store_optix = findNvOptixDll();
        cout << "ERROR: OptiX ray tracing is required, but C:/Windows/System32/nvoptix.dll was not found." << endl;
        if (!driver_store_optix.empty()) {
            cout << "A driver copy was found here: " << driver_store_optix << endl;
        }
        cout << "Run restore_optix_admin.ps1 as administrator, then run this program again." << endl;
        return 2;
    }

    bool use_optix = optix_required && !optix_dll.empty();
    if (!use_optix) {
        cout << "WARNING: OptiX is disabled by --no-optix. RadiationModel will be skipped." << endl;
        cout << "The program will use measured CSV PAR input or fallback PAR estimates and still write all output files." << endl;
    }
    if (use_optix) {
        cout << "nvoptix.dll found: " << optix_dll << endl;
        addDllSearchDirectory(optix_dll);
        try {
            radiationmodel = new RadiationModel(&context);
            radiationmodel->addRadiationBand("PAR");
            radiationmodel->setDirectRayCount("PAR", 5000);
            radiationmodel->setDiffuseRayCount("PAR", 2000);
            radiationmodel->setScatteringDepth("PAR", 2);
            radiationmodel->disableEmission("PAR");
            radiationmodel->updateGeometry();
            sun_source = radiationmodel->addCollimatedRadiationSource();
            radiation_available = true;
        } catch (const std::exception& e) {
            cout << "WARNING: RadiationModel could not be initialized: " << e.what() << endl;
            cout << "The program will use measured CSV PAR input or fallback PAR estimates." << endl;
            if (radiationmodel != nullptr) {
                delete radiationmodel;
                radiationmodel = nullptr;
            }
            radiation_available = false;
        }
    }

    EnergyBalanceModel energybalance(&context);
    energybalance.addRadiationBand("PAR");
    energybalance.addRadiationBand("LW");
    PhotosynthesisModel photosynthesis(&context);
    StomatalConductanceModel conductance(&context);
    PlantHydraulicsModel hydraulics(&context);

    for (int hour = 0; hour < 24; ++hour) {
        const MetRecord& met = hourly_met[hour];
        const float temperature_k = met.air_temperature_c + 273.15f;
        const float humidity_fraction = std::max(0.0f, std::min(1.0f, met.relative_humidity_percent / 100.0f));

        context.setTime(0, hour);
        setLeafParameters(context, all_leaves, met);
        context.setPrimitiveData(all_leaves, "radiation_flux_LW", longwaveRadiationFlux(temperature_k));

        vec3 sun_direction = solarposition.getSunDirectionVector();
        float total_flux = solarposition.getSolarFlux(
                met.air_pressure_pa,
                temperature_k,
                humidity_fraction,
                0.05f
        );
        float diffuse_fraction = solarposition.getDiffuseFraction(
                met.air_pressure_pa,
                temperature_k,
                humidity_fraction,
                0.05f
        );

        const float par_fraction = 0.45f;
        float modeled_par = std::max(0.0f, total_flux * par_fraction);
        float fallback_par = met.par_input >= 0.0f ? met.par_input : modeled_par;
        float incident_par = met.par_input >= 0.0f ? met.par_input : modeled_par;

        if (radiation_available) {
            try {
                radiationmodel->setSourcePosition(sun_source, sun_direction);
                radiationmodel->setSourceFlux(sun_source, "PAR", incident_par * (1.0f - diffuse_fraction));
                radiationmodel->setDiffuseRadiationFlux("PAR", incident_par * diffuse_fraction);
                radiationmodel->runBand("PAR");
            } catch (const std::exception& e) {
                cout << "WARNING: RadiationModel failed at hour " << hour << ": " << e.what() << endl;
                radiation_available = false;
                context.setPrimitiveData(all_leaves, "radiation_flux_PAR", fallback_par);
            }
        } else {
            context.setPrimitiveData(all_leaves, "radiation_flux_PAR", fallback_par);
        }

        // Greenhouse wall/roof visual data assignment is commented out.
        if (!greenhouse_wall.empty()) {
            context.setPrimitiveData(greenhouse_wall, "visual_temperature_c", met.air_temperature_c);
            context.setPrimitiveData(greenhouse_wall, "visual_rh_leaf_percent", met.relative_humidity_percent);
            context.setPrimitiveData(greenhouse_wall, "visual_wet_state", met.relative_humidity_percent >= 95.0f ? 1.0f : 0.0f);
            context.setPrimitiveData(greenhouse_wall, "visual_par", fallback_par);
        }
        if (!greenhouse_floor.empty()) {
            context.setPrimitiveData(greenhouse_floor, "visual_temperature_c", met.air_temperature_c + 0.8f);
            context.setPrimitiveData(greenhouse_floor, "visual_rh_leaf_percent", std::max(0.0f, met.relative_humidity_percent - 3.0f));
            context.setPrimitiveData(greenhouse_floor, "visual_wet_state", met.relative_humidity_percent >= 95.0f ? 1.0f : 0.0f);
            context.setPrimitiveData(greenhouse_floor, "visual_par", 0.82f * fallback_par);
        }

        if (radiation_available) {
            try {
                energybalance.run(all_leaves);
                hydraulics.run(all_leaves);
                photosynthesis.run();
                conductance.run();
            } catch (const std::exception& e) {
                cout << "WARNING: physiology model failed at hour " << hour << ": " << e.what() << endl;
            }
        }

        const float es_air = tetensPa(met.air_temperature_c);
        const float ea = es_air * humidity_fraction;

        for (auto& plant_pair : plant_leaf_data) {
            vector<uint>& leaves = plant_prims[plant_pair.first];
            for (size_t i = 0; i < leaves.size(); ++i) {
                LeafData& leaf = plant_pair.second[i];
                uint leaf_id = leaves[i];

                float par = 0.0f;
                if (context.doesPrimitiveDataExist(leaf_id, "radiation_flux_PAR")) {
                    context.getPrimitiveData(leaf_id, "radiation_flux_PAR", par);
                }

                float leaf_temperature_k = temperature_k;
                if (context.doesPrimitiveDataExist(leaf_id, "temperature")) {
                    context.getPrimitiveData(leaf_id, "temperature", leaf_temperature_k);
                }
                if (!radiation_available) {
                    leaf_temperature_k = temperature_k + par * 0.015f;
                    context.setPrimitiveData(leaf_id, "temperature", leaf_temperature_k);
                }
                float leaf_temperature_c = leaf_temperature_k - 273.15f;

                float es_leaf = tetensPa(leaf_temperature_c);
                float rh_leaf = es_leaf > 1e-6f ? (ea / es_leaf) * 100.0f : 0.0f;
                rh_leaf = std::max(0.0f, std::min(100.0f, rh_leaf));

                int wet_state = 0;
                if (rh_leaf >= 100.0f) {
                    wet_state = 2;
                } else if (rh_leaf >= 95.0f) {
                    wet_state = 1;
                }

                leaf.par_values[hour] = par;
                leaf.temperature_values[hour] = leaf_temperature_c;
                leaf.rh_leaf_values[hour] = rh_leaf;
                leaf.wet_state_values[hour] = wet_state;

                context.setPrimitiveData(leaf_id, "visual_par", par);
                context.setPrimitiveData(leaf_id, "visual_temperature_c", leaf_temperature_c);
                context.setPrimitiveData(leaf_id, "visual_rh_leaf_percent", rh_leaf);
                context.setPrimitiveData(leaf_id, "visual_wet_state", static_cast<float>(wet_state));
            }
        }

        cout << "Completed hour " << hour
             << " Tair(C)=" << met.air_temperature_c
             << " RH(%)=" << met.relative_humidity_percent
             << " wind=" << met.wind_speed
             << " pressure(Pa)=" << met.air_pressure_pa
             << " PAR_input=" << met.par_input << endl;
    }

    writeTimeseries(base_path + "PAR_timeseries.txt", "t", time_hours, plant_leaf_data, "PAR");
    writeTimeseries(base_path + "Temperature_timeseries.txt", "t", time_hours, plant_leaf_data, "TLEAF");
    writeTimeseries(base_path + "RHLeaf_timeseries.txt", "t", time_hours, plant_leaf_data, "RHLEAF");
    writeTimeseries(base_path + "WetState_timeseries.txt", "t", time_hours, plant_leaf_data, "WET");

    vector<string> hour_labels = buildHourLabels(time_hours);
    HourlySeriesSummary par_summary = summarizeLeafMetric(
            plant_leaf_data, time_hours, "PAR", "Radiation Flux PAR", "W m-2"
    );
    HourlySeriesSummary temperature_summary = summarizeLeafMetric(
            plant_leaf_data, time_hours, "TLEAF", "Leaf Temperature", "C"
    );
    HourlySeriesSummary rh_leaf_summary = summarizeLeafMetric(
            plant_leaf_data, time_hours, "RHLEAF", "Leaf Relative Humidity", "%"
    );
    HourlySeriesSummary wet_summary = summarizeLeafMetric(
            plant_leaf_data, time_hours, "WET", "Leaf Wet State", "class"
    );

    const size_t snapshot_hour_index = std::min<size_t>(12, hourly_met.empty() ? size_t(0) : hourly_met.size() - 1);
    const MetRecord& snapshot_met = hourly_met[snapshot_hour_index];

    vec2 wall_xbounds, wall_ybounds, wall_zbounds;
    context.getDomainBoundingBox(all_leaves, wall_xbounds, wall_ybounds, wall_zbounds);
    float floor_width = (wall_xbounds.y - wall_xbounds.x) + 1.0f;
    float floor_length = (wall_ybounds.y - wall_ybounds.x) + 2.0f;
    float floor_z = wall_zbounds.x - 0.02f;

    // Floor creation is commented out; greenhouse_floor stays empty.
    greenhouse_floor.push_back(
            context.addPatch(
                    make_vec3(0.5f * (wall_xbounds.x + wall_xbounds.y), 0.5f * (wall_ybounds.x + wall_ybounds.y), floor_z),
                    make_vec2(floor_width, floor_length),
                    make_SphericalCoord(0.0f, 0.0f),
                    RGBcolor(180, 180, 180)
            )
    );
    context.setPrimitiveData(greenhouse_floor, "reflectivity_PAR", 0.18f);
    context.setPrimitiveData(greenhouse_floor, "transmissivity_PAR", 0.0f);
    context.setPrimitiveData(greenhouse_floor, "emissivity_PAR", 0.82f);
    context.setPrimitiveData(greenhouse_floor, "emissivity_LW", 0.95f);

    vector<uint> scene_prims = all_leaves;
    scene_prims.insert(scene_prims.end(), greenhouse_wall.begin(), greenhouse_wall.end());
    scene_prims.insert(scene_prims.end(), greenhouse_floor.begin(), greenhouse_floor.end());

    vec2 xbounds, ybounds, zbounds;
    context.getDomainBoundingBox(scene_prims, xbounds, ybounds, zbounds);
    vec3 scene_center = make_vec3(
            0.5f * (xbounds.x + xbounds.y),
            0.5f * (ybounds.x + ybounds.y),
            0.5f * (zbounds.x + zbounds.y)
    );
    float span_x = xbounds.y - xbounds.x;
    float span_y = ybounds.y - ybounds.x;
    float span_z = zbounds.y - zbounds.x;
    float camera_radius = 0.82f * std::max(span_x, std::max(span_y, span_z));
    vec3 camera_look_at = make_vec3(scene_center.x, scene_center.y, zbounds.x + 0.28f * span_z);

    const string page_index = base_path + "visualization_index.html";
    const string page_meteorology = base_path + "visual_meteorology.html";
    const string page_par = base_path + "visual_par.html";
    const string page_temperature = base_path + "visual_temperature.html";
    const string page_rhleaf = base_path + "visual_rhleaf.html";
    const string page_wetstate = base_path + "visual_wetstate.html";
    const string page_disease = base_path + "visual_disease.html";
    const string image_par = base_path + "scene_par.jpg";
    const string image_temperature = base_path + "scene_temperature.jpg";
    const string image_rhleaf = base_path + "scene_rhleaf.jpg";
    const string image_wetstate = base_path + "scene_wetstate.jpg";

    // Build a 12:00 scene snapshot with per-primitive values for spatial visualization.
    setLeafParameters(context, all_leaves, snapshot_met);
    context.setPrimitiveData(all_leaves, "radiation_flux_LW", longwaveRadiationFlux(snapshot_met.air_temperature_c + 273.15f));

    bool snapshot_uses_optix = false;
    float snapshot_par_fallback = snapshot_met.par_input >= 0.0f ? snapshot_met.par_input : 0.0f;
    if (!optix_dll.empty()) {
        addDllSearchDirectory(optix_dll);
        try {
            RadiationModel snapshot_radiation(&context);
            snapshot_radiation.addRadiationBand("PAR");
            snapshot_radiation.setDirectRayCount("PAR", 6000);
            snapshot_radiation.setDiffuseRayCount("PAR", 2500);
            snapshot_radiation.setScatteringDepth("PAR", 3);
            snapshot_radiation.disableEmission("PAR");
            snapshot_radiation.updateGeometry();
            uint snapshot_source = snapshot_radiation.addCollimatedRadiationSource();

            context.setTime(0, static_cast<int>(snapshot_hour_index));
            vec3 snapshot_sun_direction = solarposition.getSunDirectionVector();
            float snapshot_temperature_k = snapshot_met.air_temperature_c + 273.15f;
            float snapshot_humidity_fraction = std::max(0.0f, std::min(1.0f, snapshot_met.relative_humidity_percent / 100.0f));
            float snapshot_total_flux = solarposition.getSolarFlux(
                    snapshot_met.air_pressure_pa,
                    snapshot_temperature_k,
                    snapshot_humidity_fraction,
                    0.05f
            );
            float snapshot_diffuse_fraction = solarposition.getDiffuseFraction(
                    snapshot_met.air_pressure_pa,
                    snapshot_temperature_k,
                    snapshot_humidity_fraction,
                    0.05f
            );
            if (snapshot_par_fallback <= 0.0f) {
                snapshot_par_fallback = std::max(0.0f, snapshot_total_flux * 0.45f);
            }

            snapshot_radiation.setSourcePosition(snapshot_source, snapshot_sun_direction);
            snapshot_radiation.setSourceFlux(snapshot_source, "PAR", snapshot_par_fallback * (1.0f - snapshot_diffuse_fraction));
            snapshot_radiation.setDiffuseRadiationFlux("PAR", snapshot_par_fallback * snapshot_diffuse_fraction);
            snapshot_radiation.runBand("PAR");
            energybalance.run(all_leaves);
            snapshot_uses_optix = true;
            cout << "12:00 scene visualization uses per-primitive OptiX radiation results." << endl;
        } catch (const std::exception& e) {
            cout << "WARNING: 12:00 scene snapshot failed, falling back to geometry-based per-primitive approximation: " << e.what() << endl;
        }
    }

    for (const auto& plant_pair : plant_prims) {
        const vector<uint>& leaves = plant_pair.second;
        for (uint leaf_id : leaves) {
            float par = 0.0f;
            float leaf_temperature_c = snapshot_met.air_temperature_c;
            float rh_leaf = snapshot_met.relative_humidity_percent;
            int wet_state = 0;

            if (snapshot_uses_optix && context.doesPrimitiveDataExist(leaf_id, "radiation_flux_PAR")) {
                context.getPrimitiveData(leaf_id, "radiation_flux_PAR", par);
                float leaf_temperature_k = snapshot_met.air_temperature_c + 273.15f;
                if (context.doesPrimitiveDataExist(leaf_id, "temperature")) {
                    context.getPrimitiveData(leaf_id, "temperature", leaf_temperature_k);
                }
                leaf_temperature_c = leaf_temperature_k - 273.15f;
                float es_air = tetensPa(snapshot_met.air_temperature_c);
                float ea = es_air * std::max(0.0f, std::min(1.0f, snapshot_met.relative_humidity_percent / 100.0f));
                float es_leaf = tetensPa(leaf_temperature_c);
                rh_leaf = es_leaf > 1e-6f ? std::max(0.0f, std::min(100.0f, (ea / es_leaf) * 100.0f)) : snapshot_met.relative_humidity_percent;
                wet_state = rh_leaf >= 100.0f ? 2 : (rh_leaf >= 95.0f ? 1 : 0);
            } else {
                vec3 min_corner, max_corner;
                context.getPrimitiveBoundingBox(leaf_id, min_corner, max_corner);
                vec3 normal = context.getPrimitiveNormal(leaf_id);
                float leaf_height = 0.5f * (min_corner.z + max_corner.z);
                float normalized_height = span_z > 1e-6f ? std::max(0.0f, std::min(1.0f, (leaf_height - zbounds.x) / span_z)) : 0.5f;
                float orientation_factor = std::max(0.15f, fabs(normal.z));
                float row_variation = 0.85f + 0.15f * sinf(0.35f * min_corner.y + 1.4f * min_corner.x);
                float side_variation = 0.88f + 0.12f * cosf(2.0f * min_corner.x);
                par = snapshot_par_fallback * (0.30f + 0.70f * normalized_height * orientation_factor) * row_variation * side_variation;
                par = std::max(0.0f, par);
                leaf_temperature_c = snapshot_met.air_temperature_c + 0.020f * par - 2.2f * (1.0f - normalized_height);
                rh_leaf = std::max(0.0f, std::min(100.0f, snapshot_met.relative_humidity_percent - 0.65f * (leaf_temperature_c - snapshot_met.air_temperature_c)));
                wet_state = rh_leaf >= 100.0f ? 2 : (rh_leaf >= 95.0f ? 1 : 0);
            }

            context.setPrimitiveData(leaf_id, "scene_visual_par", par);
            context.setPrimitiveData(leaf_id, "scene_visual_temperature_c", leaf_temperature_c);
            context.setPrimitiveData(leaf_id, "scene_visual_rh_leaf_percent", rh_leaf);
            context.setPrimitiveData(leaf_id, "scene_visual_wet_state", static_cast<float>(wet_state));
        }
    }

    // Greenhouse structure visual data for the snapshot is commented out.
    if (!greenhouse_wall.empty()) {
        context.setPrimitiveData(greenhouse_wall, "scene_visual_par", 0.75f * snapshot_par_fallback);
        context.setPrimitiveData(greenhouse_wall, "scene_visual_temperature_c", snapshot_met.air_temperature_c);
        context.setPrimitiveData(greenhouse_wall, "scene_visual_rh_leaf_percent", snapshot_met.relative_humidity_percent);
        context.setPrimitiveData(greenhouse_wall, "scene_visual_wet_state", snapshot_met.relative_humidity_percent >= 95.0f ? 1.0f : 0.0f);
    }
    if (!greenhouse_floor.empty()) {
        context.setPrimitiveData(greenhouse_floor, "scene_visual_par", 0.82f * snapshot_par_fallback);
        context.setPrimitiveData(greenhouse_floor, "scene_visual_temperature_c", snapshot_met.air_temperature_c + 0.8f);
        context.setPrimitiveData(greenhouse_floor, "scene_visual_rh_leaf_percent", std::max(0.0f, snapshot_met.relative_humidity_percent - 3.0f));
        context.setPrimitiveData(greenhouse_floor, "scene_visual_wet_state", snapshot_met.relative_humidity_percent >= 95.0f ? 1.0f : 0.0f);
    }

    writeMeteorologyVisualizationPage(page_meteorology, hourly_met);
    writeVisualizationPage(
            page_par,
            "PAR Visualization",
            hour_labels,
            par_summary,
            "line",
            "#2563eb",
            "Hourly mean, minimum and maximum PAR values across all loaded cucumber leaf primitives."
    );
    writeVisualizationPage(
            page_temperature,
            "Leaf Temperature Visualization",
            hour_labels,
            temperature_summary,
            "line",
            "#dc2626",
            "Hourly mean, minimum and maximum leaf temperature across all loaded cucumber leaf primitives."
    );
    writeVisualizationPage(
            page_rhleaf,
            "Leaf Relative Humidity Visualization",
            hour_labels,
            rh_leaf_summary,
            "line",
            "#0891b2",
            "Hourly mean, minimum and maximum leaf relative humidity across all loaded cucumber leaf primitives."
    );
    writeVisualizationPage(
            page_wetstate,
            "Leaf Wet State Visualization",
            hour_labels,
            wet_summary,
            "line",
            "#7c3aed",
            "Hourly mean, minimum and maximum wetness class across all loaded cucumber leaf primitives."
    );
    writeDiseaseVisualizationPage(page_disease, disease_metrics);
    exportSceneVisualizationJPEG(
            context,
            all_leaves,
            scene_prims,
            "scene_visual_par",
            "Solar radiation intensity (W m-2)",
            image_par,
            Visualizer::COLORMAP_RAINBOW,
            camera_look_at,
            camera_radius
    );
    exportSceneVisualizationJPEG(
            context,
            all_leaves,
            scene_prims,
            "scene_visual_temperature_c",
            "Temperature (C)",
            image_temperature,
            Visualizer::COLORMAP_RAINBOW,
            camera_look_at,
            camera_radius
    );
    exportSceneVisualizationJPEG(
            context,
            all_leaves,
            scene_prims,
            "scene_visual_rh_leaf_percent",
            "Relative humidity (%)",
            image_rhleaf,
            Visualizer::COLORMAP_RAINBOW,
            camera_look_at,
            camera_radius
    );
    exportSceneVisualizationJPEG(
            context,
            all_leaves,
            scene_prims,
            "scene_visual_wet_state",
            "Leaf wet state",
            image_wetstate,
            Visualizer::COLORMAP_RAINBOW,
            camera_look_at,
            camera_radius
    );
    writeVisualizationIndexPage(
            page_index,
            {
                    {"Meteorology Overview", "visual_meteorology.html"},
                    {"PAR Visualization", "visual_par.html"},
                    {"Leaf Temperature Visualization", "visual_temperature.html"},
                    {"Leaf Relative Humidity Visualization", "visual_rhleaf.html"},
                    {"Leaf Wet State Visualization", "visual_wetstate.html"},
                    {"Disease Risk Visualization", "visual_disease.html"},
                    {"Scene PAR Image", "scene_par.jpg"},
                    {"Scene Temperature Image", "scene_temperature.jpg"},
                    {"Scene RH Image", "scene_rhleaf.jpg"},
                    {"Scene Wet State Image", "scene_wetstate.jpg"}
            }
    );

    cout << "Simulation completed." << endl;
    cout << "Files generated:" << endl;
    cout << "- " << base_path + "PAR_timeseries.txt" << endl;
    cout << "- " << base_path + "Temperature_timeseries.txt" << endl;
    cout << "- " << base_path + "RHLeaf_timeseries.txt" << endl;
    cout << "- " << base_path + "WetState_timeseries.txt" << endl;
    cout << "- " << base_path + "Disease_5min_Result.txt" << endl;
    cout << "- " << base_path + "Disease_Daily_Result.txt" << endl;
    cout << "- " << page_index << endl;
    cout << "- " << page_meteorology << endl;
    cout << "- " << page_par << endl;
    cout << "- " << page_temperature << endl;
    cout << "- " << page_rhleaf << endl;
    cout << "- " << page_wetstate << endl;
    cout << "- " << page_disease << endl;
    cout << "- " << image_par << endl;
    cout << "- " << image_temperature << endl;
    cout << "- " << image_rhleaf << endl;
    cout << "- " << image_wetstate << endl;
    cout << "Scene inputs used:" << endl;
    cout << "- Plant directory: " << ply_dir << endl;
    cout << "- Greenhouse wall: " << greenhouse_wall_ply << endl;
    cout << "- Greenhouse roof: not used" << endl;
    cout << "- Greenhouse floor: generated patch from canopy bounding box" << endl;
    cout << "Runtime mode:" << endl;
    cout << "- OptiX " << (optix_required ? "enabled" : "disabled by default") << endl;
    cout << "- Use --use-optix to enable ray tracing when needed." << endl;
    cout << "- Scene snapshot time = 2026-06-05 12:00" << endl;
    cout << "- Visualization " << (visualization_enabled ? "enabled by default" : "disabled") << endl;
    cout << "- Use --no-vis to skip the interactive 12:00 viewers." << endl;
    if (visualization_enabled) {
        showSceneVisualizationInteractive(
                context, all_leaves, scene_prims, "scene_visual_par", "Solar radiation intensity (W m-2) at 12:00",
                Visualizer::COLORMAP_RAINBOW, camera_look_at, camera_radius
        );
        showSceneVisualizationInteractive(
                context, all_leaves, scene_prims, "scene_visual_temperature_c", "Temperature (C) at 12:00",
                Visualizer::COLORMAP_RAINBOW, camera_look_at, camera_radius
        );
        showSceneVisualizationInteractive(
                context, all_leaves, scene_prims, "scene_visual_rh_leaf_percent", "Relative humidity (%) at 12:00",
                Visualizer::COLORMAP_RAINBOW, camera_look_at, camera_radius
        );
        showSceneVisualizationInteractive(
                context, all_leaves, scene_prims, "scene_visual_wet_state", "Leaf wet state at 12:00",
                Visualizer::COLORMAP_RAINBOW, camera_look_at, camera_radius
        );
    }
    if (radiationmodel != nullptr) {
        delete radiationmodel;
    }
    return 0;
}
