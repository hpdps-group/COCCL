#include "misc/tuning/coccl_autotune_internal.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

cocclOperation operationFromName(const std::string& name) {
  return name == "reducescatter"
      ? cocclOperation::ReduceScatter
      : cocclOperation::AllReduce;
}

std::string candidateNames(const cocclAutotuneCandidateSet& candidates) {
  std::string result;
  for (size_t i = 0; i < candidates.count; ++i) {
    if (!result.empty()) result += ';';
    result += candidates.candidates[i].spec->name;
  }
  return result;
}

double scoreForSlot(const cocclAutotuneCandidateSet& candidates,
                    cocclAutotuneScoreSlot slot) {
  for (size_t i = 0; i < candidates.count; ++i) {
    if (candidates.candidates[i].spec->scoreSlot == slot) {
      return candidates.candidates[i].scoreUs;
    }
  }
  return std::numeric_limits<double>::infinity();
}

}  // namespace

int main() {
  std::string line;
  std::getline(std::cin, line);
  std::cout
      << "case_id,operation,compressor,bytes,eligible_candidates,oneshot_us,twoshot_us,tripleshot_us,selected_algorithm,used_model\n";
  std::cout << std::setprecision(17);
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = split(line);
    if (fields.size() != 16) return 2;

    const std::string& caseId = fields[0];
    const std::string& operationName = fields[1];
    const std::string& compressor = fields[2];
    const double bytes = std::strtod(fields[3].c_str(), nullptr);
    const int localRanks = std::atoi(fields[4].c_str());
    const int nodes = std::atoi(fields[5].c_str());
    const cocclSelectionPerformanceModel performance = {
        {std::strtod(fields[6].c_str(), nullptr),
         std::strtod(fields[7].c_str(), nullptr), true},
        {std::strtod(fields[8].c_str(), nullptr),
         std::strtod(fields[9].c_str(), nullptr), true},
    };
    const cocclCodecModel baseCodec = {
        {std::strtod(fields[10].c_str(), nullptr),
         std::strtod(fields[11].c_str(), nullptr), true},
        std::strtod(fields[12].c_str(), nullptr), true};
    const cocclCodecModel hierarchicalCodec = {
        {std::strtod(fields[13].c_str(), nullptr),
         std::strtod(fields[14].c_str(), nullptr), true},
        std::strtod(fields[15].c_str(), nullptr), true};

    const cocclOperation operation = operationFromName(operationName);
    const cocclAutotuneEligibility eligibility = {
        nodes > 1 && localRanks > 1, true};
    cocclAutotuneCandidateSet candidates =
        cocclAutotuneBuildCandidates(operation, eligibility);
    const cocclAutotuneCodecSet codecs = {
        {true, &baseCodec},
        {true, &hierarchicalCodec},
        {true, &hierarchicalCodec},
    };
    for (size_t i = 0; i < candidates.count; ++i) {
      cocclAutotuneCandidate& candidate = candidates.candidates[i];
      candidate.scoreUs = cocclAutotuneEvaluateCost(
          candidate.spec->costKind, performance, codecs, bytes,
          localRanks, nodes);
    }
    const cocclAutotuneDecision decision = cocclAutotuneChooseCandidate(
        candidates, cocclAlgorithmNone, true);
    if (decision.candidate == nullptr) return 3;

    std::cout << caseId << ',' << operationName << ',' << compressor << ','
              << (uint64_t)bytes << ',' << candidateNames(candidates) << ','
              << scoreForSlot(candidates, cocclAutotuneScoreSlot::OneShot)
              << ','
              << scoreForSlot(candidates, cocclAutotuneScoreSlot::TwoShot)
              << ','
              << scoreForSlot(candidates, cocclAutotuneScoreSlot::TripleShot)
              << ',' << decision.candidate->name << ','
              << (int)decision.usedModel << '\n';
  }
  return 0;
}
