#include "util.h"
#include <log/log.h>
#include <sys/time.h>

static double _start_time[256];
static int _p = -1;

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start() { _start_time[++_p] = get_time(); }

double timer_stop() { return get_time() - _start_time[_p--]; }

void timer_stop_and_log(const char *name) { log_debug("%s: %lf", name, timer_stop()); }
