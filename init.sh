source /home/openvino-ci-58/tingqian/venv/bin/activate
source /home/openvino-ci-58/intel/oneapi/setvars.sh

function buildops
{
    cmake --build ops/build/ --config Release --verbose
}