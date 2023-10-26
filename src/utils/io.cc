
#include "io.h"

namespace io {
bool IsSimu(timestamp_t t1, timestamp_t t2)//interval 5000000
{
    if(std::abs(long(t1 - t2)) < 25000000)
        return true;
    return false;
}
// ------------------------------------------------------------

} // namespace io
