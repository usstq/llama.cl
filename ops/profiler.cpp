// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "profiler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static uint64_t tsc_ticks_per_second;
static uint64_t tsc_ticks_base;
inline uint64_t tsc_to_usec(uint64_t tsc_ticks) {
  return (tsc_ticks - tsc_ticks_base) * 1000000 / tsc_ticks_per_second;
}
inline uint64_t tsc_to_usec(uint64_t tsc_ticks0, uint64_t tsc_ticks1) {
  return (tsc_ticks1 - tsc_ticks0) * 1000000 / tsc_ticks_per_second;
}
static void init_tsc() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    uint64_t start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    tsc_ticks_per_second = (__rdtsc() - start_ticks);
    std::cout << "[OPS_PROFILE] tsc_ticks_per_second = " << tsc_ticks_per_second
              << std::endl;
    tsc_ticks_base = __rdtsc();
  });
}

// to minimize the overhead of profiler when it's not being enabled,
// the inst is not put inside a singleton function to save extra
// cost in multi-threading safety checks.

struct chromeTrace {
  std::ostream& os;
  int fake_tid;
  uint64_t ts;
  chromeTrace(std::ostream& os, int fake_tid) : os(os), fake_tid(fake_tid) {}
  void addCompleteEvent(std::string name,
                        std::string cat,
                        uint64_t start,
                        uint64_t dur) {
    // chrome tracing will show & group-by to name, so we use cat as name
    os << "{\"ph\": \"X\", \"name\": \"" << cat << "\", \"cat\":\"" << name
       << "\","
       << "\"pid\": " << fake_tid << ", \"tid\": 0,"
       << "\"ts\": " << start << ", \"dur\": " << dur << "},\n";
  }
};

thread_local ProfilerManager profilerManagerInstance;

static std::atomic_int totalProfilerManagers{0};

void ProfileData::record_end(ProfileData* p) {
  p->end = __rdtsc();
}

bool not_finalized = true;

struct ProfilerManagerFinalizer {
  std::mutex g_mutex;
  std::set<ProfilerManager*> all_managers;
  const char* dump_file_name = "ops_profile.json";
  bool dump_file_over = false;

  ~ProfilerManagerFinalizer() {
    if (not_finalized)
      finalize();
  }

  void finalize() {
    if (!not_finalized)
      return;

    std::lock_guard<std::mutex> guard(g_mutex);
    if (dump_file_over || all_managers.empty())
      return;

    // start dump
    std::ofstream fw;
    fw.open(dump_file_name, std::ios::out);
    fw << "{\n";
    fw << "\"schemaVersion\": 1,\n";
    fw << "\"traceEvents\": [\n";
    fw.flush();

    for (auto& pthis : all_managers) {
      if (!pthis->enabled)
        continue;
      auto data_size = pthis->all_data.size();
      if (!data_size)
        continue;

      // open output file
      std::map<std::string, uint64_t> cat_us;
      std::map<std::string, uint64_t> cat_cnt;
      chromeTrace ct(fw, pthis->serial);
      for (auto& d : pthis->all_data) {
        auto duration = tsc_to_usec(d.start, d.end);
        if (cat_us.count(d.cat) == 0) {
          cat_us[d.cat] = 0;
          cat_cnt[d.cat] = 0;
        }
        cat_us[d.cat] += duration;
        cat_cnt[d.cat]++;
        ct.addCompleteEvent(d.name, d.cat, tsc_to_usec(d.start), duration);
      }
      pthis->all_data.clear();
      std::cout << "[OPS_PROFILE] #" << pthis->serial << "(" << pthis
                << ") finalize: dumpped " << data_size << std::endl;

      std::cout << "[OPS_PROFILE] Summary" << std::endl;
      for (auto& kv : cat_cnt) {
        auto& key = kv.first;
        std::cout << std::setw(16) << key << " : " << std::setw(16)
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(cat_us[key]) / cat_cnt[key] << " us x "
                  << cat_cnt[key] << std::endl;
      }
    }
    all_managers.clear();

    fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
       << tsc_to_usec(__rdtsc()) << "}",
        fw << "]\n";
    fw << "}\n";
    auto total_size = fw.tellp();
    fw.close();
    dump_file_over = true;
    not_finalized = false;
    std::cout << "[OPS_PROFILE] Dumpped " << total_size / (1024 * 1024)
              << " (MB) to " << dump_file_name << std::endl;
  }

  void register_manager(ProfilerManager* pthis) {
    std::lock_guard<std::mutex> guard(g_mutex);
    std::stringstream ss;
    ss << "[OPS_PROFILE] #" << pthis->serial << "(" << this
       << ") : is registed." << std::endl;
    std::cout << ss.str();
    all_managers.emplace(pthis);
  }
} finalizer;

ProfilerManager::ProfilerManager() {
  const char* str_enable = std::getenv("OPS_PROFILE");
  if (!str_enable)
    str_enable = "0";
  enabled = atoi(str_enable) > 0;
  if (enabled) {
    init_tsc();
    tid = std::this_thread::get_id();
    serial = totalProfilerManagers.fetch_add(1);
    finalizer.register_manager(this);
  }
}

ProfilerManager::~ProfilerManager() {
  if (not_finalized)
    finalizer.finalize();
}
