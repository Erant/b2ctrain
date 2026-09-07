#include "train/trainer.h"
#include "dataset/views.h"
#include "util/log.h"
namespace b2c {
int train_main(const Config& cfg) {
  Dataset ds = load_dataset(cfg);
  log_info("training not implemented yet (%zu views)", ds.train.size());
  return 0;
}
}
