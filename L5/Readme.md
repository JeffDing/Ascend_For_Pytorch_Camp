# Ascend for Pytroch第五课

# 环境准备

## 安装torch\-2\.7\.1

```Bash
pip install torch==2.7.1 torchvision
```

## 克隆Torch\-NPU代码仓

```Bash
git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-26.0.0 --recursive
cd pytorch
```

## 设置MAX\_JOBS

```Bash
export MAX_JOBS=100
```

# 实践部分

## 题目一：在 NPUGuard 中插入日志，验证 RAII 行为

**目标文件**：`pytorch_npu/torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp`

**任务**：

1. 在 `exchangeDevice` / `setDevice` / `uncheckedSetDevice` / `getDevice` 中插入日志

2. 重新编译 torch\_npu

3. 运行 `x.to("npu")`，观察日志成对出现（切出/还原）

**验证**：日志



## 操作部分：

### 修改`pytorch_npu/torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp`

完整代码部分

```C++
#pragma GCC visibility push(default)
#include <chrono>
#include <thread>
#include <torch/csrc/jit/serialization/pickler.h>
#include "torch_npu/csrc/core/npu/impl/NPUGuardImpl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUSerialization.h"
#include "torch_npu/csrc/core/npu/NPUHooksInterface.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"

#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace c10_npu {

namespace impl {

constexpr c10::DeviceType NPUGuardImpl::static_type;

NPUGuardImpl::NPUGuardImpl(c10::DeviceType t)
{
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1, "DeviceType must be NPU. Actual DeviceType is: ", t,
                          PTA_ERROR(ErrCode::PARAM));
}

c10::Device NPUGuardImpl::exchangeDevice(c10::Device d) const
{
    ASCEND_LOGI("[NPUGuard] exchangeDevice enter, target_dev_idx=%d", d.index());
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", d.type(), PTA_ERROR(ErrCode::PARAM));
    c10::Device old_device = getDevice();
    ASCEND_LOGI("[NPUGuard] exchangeDevice get old_dev=%d, target_dev=%d", old_device.index(), d.index());
    if (old_device.index() != d.index()) {
        NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(d.index()));
        ASCEND_LOGI("[NPUGuard] exchangeDevice switch success from %d -> %d", old_device.index(), d.index());
    } else {
        ASCEND_LOGI("[NPUGuard] exchangeDevice skip switch, old == target dev %d", d.index());
    }
    ASCEND_LOGI("[NPUGuard] exchangeDevice exit, return saved_old_dev=%d", old_device.index());
    return old_device;
}

c10::Device NPUGuardImpl::getDevice() const
{
    int device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&device));
    ASCEND_LOGI("[NPUGuard] getDevice query current dev=%d", device);
    return c10::Device(c10::DeviceType::PrivateUse1, device);
}

void NPUGuardImpl::setDevice(c10::Device d) const
{
    ASCEND_LOGI("[NPUGuard] setDevice enter, target_dev=%d", d.index());
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", d.type(), PTA_ERROR(ErrCode::PARAM));
    NPU_CHECK_ERROR(c10_npu::SetDevice(d.index()));
    ASCEND_LOGI("[NPUGuard] setDevice exit, set dev=%d done", d.index());
}

void NPUGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept
{
    ASCEND_LOGI("[NPUGuard] uncheckedSetDevice enter, target_dev=%d", d.index());
    c10_npu::StartMainThreadBind(d.index());
    NPU_CHECK_WARN(c10_npu::MaybeSetDevice(d.index()));
    ASCEND_LOGI("[NPUGuard] uncheckedSetDevice exit, dev=%d set done", d.index());
}

c10::Stream NPUGuardImpl::getStream(c10::Device d) const noexcept
{
    return c10_npu::getCurrentNPUStream(d.index()).unwrap();
}

c10::Stream NPUGuardImpl::getDefaultStream(c10::Device d) const
{
    return c10_npu::getDefaultNPUStream(d.index());
}

c10::Stream NPUGuardImpl::getNewStream(c10::Device d, int priority) const
{
    bool isHighPriority = priority != 0 ? true : false;
    return c10_npu::getStreamFromPool(isHighPriority, d.index());
}

c10::Stream NPUGuardImpl::getStreamFromGlobalPool(c10::Device d, bool isHighPriority) const
{
    return c10_npu::getStreamFromPool(isHighPriority, d.index());
}

c10::Stream NPUGuardImpl::exchangeStream(c10::Stream s) const noexcept
{
    NPUStream cs(s);
    auto old_stream = c10_npu::getCurrentNPUStream(s.device().index());
    c10_npu::setCurrentNPUStream(cs);
    return old_stream.unwrap();
}

c10::DeviceIndex NPUGuardImpl::deviceCount() const noexcept
{
    static c10::DeviceIndex count = c10_npu::device_count();
    return count;
}

// Event-related functions
void NPUGuardImpl::createEvent(aclrtEvent *acl_event, const c10::EventFlag flag) const
{
    auto flag_ = ACL_EVENT_DEFAULT;
    if (c10_npu::acl::IsExistCreateEventExWithFlag()) {
        // BACKEND_DEFAULT is a enable-timing flag
        flag_ = (flag == c10::EventFlag::BACKEND_DEFAULT) ? (ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) : ACL_EVENT_SYNC;
    } else {
        flag_ = (flag == c10::EventFlag::BACKEND_DEFAULT) ? ACL_EVENT_TIME_LINE : ACL_EVENT_DEFAULT;
    }
    ASCEND_LOGI("Event: Mapped ACL event flag = 0x%x", flag_);
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtCreateEventWithFlag(acl_event, flag_));
    ASCEND_LOGI("Event: aclrtCreateEventWithFlag is successfully executed, event=%p", *acl_event);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventCreation(reinterpret_cast<uintptr_t>(*acl_event));
    }
#endif
}

void NPUGuardImpl::destroyEvent(void *event, const c10::DeviceIndex device_index) const noexcept
{
    if (!event) {
        return;
    }
    auto acl_event = static_cast<aclrtEvent>(event);
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchLazyDestroyEventTask(acl_event, device_index));
    if (!c10_npu::acl::IsExistCreateEventExWithFlag() || c10_npu::option::OptionsManager::GetPerStreamQueue()) {
        c10_npu::NPUEventManager::GetInstance().QueryAndDestroyEvent();
    }
    ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", acl_event);
}

void NPUGuardImpl::record(void **event, const c10::Stream &stream, const c10::DeviceIndex device_index,
                          const c10::EventFlag flag) const
{
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(), "Event device index ", device_index,
                " does not match recording stream's device index ", stream.device_index(), ".",
                PTA_ERROR(ErrCode::PARAM));

    aclrtEvent npu_event = static_cast<aclrtEvent>(*event);
    NPUStream npu_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!npu_event) {
        createEvent(&npu_event, flag);
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchRecordEventTask(npu_event, npu_stream, 0));
    ASCEND_LOGI("Event: aclrtRecordEvent is successfully executed, stream=%p, event=%p", npu_stream.stream(false),
                npu_event);
    // Makes the void* point to the (possibly just allocated) NPU event
    *event = npu_event;

    // Resets device
    setDevice(orig_device);
}

void NPUGuardImpl::block(void *event, const c10::Stream &stream) const
{
    if (!event) {
        return;
    }
    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    NPUStream npu_stream{stream};
    // If using multiple task queues, it is necessary to ensure that the enqueued record is dequeued before wait.
    c10_npu::NPUEventManager &mgr = c10_npu::NPUEventManager::GetInstance();
    while (c10_npu::option::OptionsManager::GetPerStreamQueue() && !mgr.IsEventRecorded(npu_event)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10)); // 10 us
    }
    const auto orig_device = getDevice();
    setDevice(stream.device());
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::queue::LaunchWaitEventTask(npu_event, npu_stream, 0));
    ASCEND_LOGI("Event: aclrtStreamWaitEvent is successfully executed, stream=%p, event=%p",
                npu_stream.stream(false), npu_event);
    setDevice(orig_device);
}

// May be called from any device
bool NPUGuardImpl::queryEvent(void *event) const
{
    if (!event) {
        return true;
    }
    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable() != 0 &&
        !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(npu_event)) {
        return false;
    }
    acl::aclrtEventRecordedStatus status = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    NPU_CHECK_ERROR_WITHOUT_UCE(acl::AclQueryEventRecordedStatus(npu_event, &status));
    return (status == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE);
}

// Stream-related functions
bool NPUGuardImpl::queryStream(const c10::Stream& stream) const
{
    NPUStream npu_stream{stream};
    return npu_stream.query();
}

void NPUGuardImpl::synchronizeStream(const c10::Stream& stream) const
{
    NPUStream npu_stream{stream};
    npu_stream.synchronize();
}

void NPUGuardImpl::synchronizeEvent(void* event) const
{
    if (!event) {
        return;
    }

    aclrtEvent npu_event = static_cast<aclrtEvent>(event);
    if (c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        c10_npu::NPUEventManager &mgr = c10_npu::NPUEventManager::GetInstance();
        while (!mgr.IsEventRecorded(npu_event)) {
        }
    }

    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtSynchronizeEvent(npu_event));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", npu_event);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(npu_event));
    }
#endif
}

void NPUGuardImpl::synchronizeDevice(const c10::DeviceIndex device_index) const
{
    int orig_device = -1;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&orig_device));
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device_index));
    c10_npu::npuSynchronizeDevice();
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(orig_device));
}

void NPUGuardImpl::recordDataPtrOnStream(const c10::DataPtr &data_ptr, const c10::Stream &stream) const
{
    NPUStream npu_stream{stream};
    c10_npu::NPUCachingAllocator::recordStream(data_ptr, npu_stream);
}

double NPUGuardImpl::elapsedTime(void *event1, void *event2, c10::DeviceIndex device_index) const
{
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    int orig_device = -1;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&orig_device));
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device_index));

    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != NPU_STATUS_SUCCESS) {
        ASCEND_LOGE("Failed to empty NPU task queue, ret: %s", ret.c_str());
    }
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event1));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event1=%p", event1);
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event2));
    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event2=%p", event2);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event1));
        trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event2));
    }
#endif
    float time_ms = 0;
    // raise error if either event is recorded but not yet completed
    NPU_CHECK_ERROR(aclrtEventElapsedTime(&time_ms, event1, event2));
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(orig_device));
    return static_cast<double>(time_ms);
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, NPUGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                                                             \
    int rename_privateuse1_backend()                                                                                   \
    {                                                                                                                  \
        c10::register_privateuse1_backend(#name);                                                                      \
        c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &torch_npu::make_npu_storage_impl);                    \
        at::RegisterPrivateUse1HooksInterface(c10_npu::get_npu_hooks());                                               \
        torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &torch_npu::npu_info_serialization,        \
                                              &torch_npu::npu_info_deserialization);                                   \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(npu)

} // namespace impl

} // namespace c10_npu
#pragma GCC visibility pop
```

### 编译

```Bash
bash ci/build.sh --python=3.12
```

### 测试脚本`test_raii_guard.py`

```Python
import torch

def test_to_npu_raii():
    print("==== Start tensor.to('npu') RAII test ====")
    cpu_t = torch.randn(2, 4)
    # 构造NPUGuard：调用exchangeDevice切到npu
    npu_t = cpu_t.to("npu:0")
    print(f"tensor device: {npu_t.device}")
    # 函数作用域结束，NPUGuard析构，再次exchangeDevice切回原始设备
    print("==== Scope exit, NPUGuard destruct trigger device restore ====")

if __name__ == "__main__":
    # 开启NPU INFO日志
    import os
    os.environ["TORCH_NPU_LOG_LEVEL"] = "INFO"
    test_to_npu_raii()
```

### 主要修改

仅修改 4 个目标函数：`exchangeDevice` / `getDevice` / `setDevice` / `uncheckedSetDevice`，其余代码完全不动，文件原有头文件、日志宏已存在无需新增。

#### c10::Device NPUGuardImpl::exchangeDevice\(c10::Device d\) const

新增 4 处日志，打印目标设备、当前旧设备、切换结果、返回保存的原始设备

```C++
c10::Device NPUGuardImpl::exchangeDevice(c10::Device d) const
{
    // 【新增日志1】函数入口，打印目标设备索引
    ASCEND_LOGI("[NPUGuard] exchangeDevice enter, target_dev_idx=%d", d.index());

    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", d.type(), PTA_ERROR(ErrCode::PARAM));
    c10::Device old_device = getDevice();

    // 【新增日志2】获取当前原始设备，打印新旧设备
    ASCEND_LOGI("[NPUGuard] exchangeDevice get old_dev=%d, target_dev=%d", old_device.index(), d.index());

    if (old_device.index() != d.index()) {
        NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(d.index()));
        // 【新增日志3】执行设备切换成功，打印切换前后设备
        ASCEND_LOGI("[NPUGuard] exchangeDevice switch success from %d -> %d", old_device.index(), d.index());
    } else {
        // 【新增日志4】新旧设备一致，无需切换
        ASCEND_LOGI("[NPUGuard] exchangeDevice skip switch, old == target dev %d", d.index());
    }

    // 【新增日志5】函数出口，打印需要保存的原始设备（RAII析构时用来还原）
    ASCEND_LOGI("[NPUGuard] exchangeDevice exit, return saved_old_dev=%d", old_device.index());
    return old_device;
}
```

#### c10::Device NPUGuardImpl::getDevice\(\) const

```Bash
c10::Device NPUGuardImpl::getDevice() const
{
    int device = 0;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&device));
    // 【新增日志】查询当前生效NPU设备编号
    ASCEND_LOGI("[NPUGuard] getDevice query current dev=%d", device);
    return c10::Device(c10::DeviceType::PrivateUse1, device);
}
```

#### void NPUGuardImpl::setDevice\(c10::Device d\) const

```C++
void NPUGuardImpl::setDevice(c10::Device d) const
{
    // 【新增日志1】setDevice函数入口，待切换设备
    ASCEND_LOGI("[NPUGuard] setDevice enter, target_dev=%d", d.index());

    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", d.type(), PTA_ERROR(ErrCode::PARAM));
    NPU_CHECK_ERROR(c10_npu::SetDevice(d.index()));

    // 【新增日志2】setDevice执行完成
    ASCEND_LOGI("[NPUGuard] setDevice exit, set dev=%d done", d.index());
}
```

#### void NPUGuardImpl::uncheckedSetDevice\(c10::Device d\) const noexcept

```C++
void NPUGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept
{
    // 【新增日志1】无校验快速切设备入口
    ASCEND_LOGI("[NPUGuard] uncheckedSetDevice enter, target_dev=%d", d.index());

    c10_npu::StartMainThreadBind(d.index());
    NPU_CHECK_WARN(c10_npu::MaybeSetDevice(d.index()));

    // 【新增日志2】无校验切设备执行完成
    ASCEND_LOGI("[NPUGuard] uncheckedSetDevice exit, dev=%d set done", d.index());
}
```



## 题目二：修改流池容量，观察 round\-robin 行为变化

**目标文件**：`pytorch_npu/torch_npu/csrc/core/npu/NPUStream.cpp`

**任务**：

1. `torch.npu.Stream()` 创建 35 个流，收集 `strea_id`，观察id变化

2. 将 `kStreamsPerPoolBits` 从 5（32）改为 3（8）

3. 重新编译

4. 创建 20 个流，收集 `strea_id`，验证轮询周期从 32 变为 8

**验证**：round\-robin 周期随 `kStreamsPerPool` 改变

### 修改代码`pytorch_npu/torch_npu/csrc/core/npu/NPUStream.cpp`

```C++
# static constexpr int kStreamsPerPoolBits = 5
修改为
static constexpr int kStreamsPerPoolBits = 3
```

### 重新编译

```C++
rm -rf build dist torch_npu.egg-info
bash ci/build.sh --python=3.12
```

### 验证脚本

#### test\_stream\_35\.py

```Python
import torch

def collect_stream_ids(count):
    id_list = []
    print(f"===== 创建 {count} 个 torch.npu.Stream()，基线 bits=5，池容量32 =====")
    for i in range(count):
        s = torch.npu.Stream()
        sid = s.stream_id  # 属性，不加()
        id_list.append(sid)
        print(f"创建序号{i:2d} | stream_id = {sid}")
    return id_list

if __name__ == "__main__":
    torch.npu.set_device(0)
    ids = collect_stream_ids(35)
    print("\n全部stream_id序列：")
    print(ids)
```

#### test\_stream\_20\.py

```Python
import torch

def collect_stream_ids(count):
    id_list = []
    print(f"===== 创建 {count} 个 torch.npu.Stream()，修改后 bits=3，池容量8 =====")
    for i in range(count):
        s = torch.npu.Stream()
        sid = s.stream_id
        id_list.append(sid)
        print(f"创建序号{i:2d} | stream_id = {sid}")
    return id_list

if __name__ == "__main__":
    torch.npu.set_device(0)
    ids = collect_stream_ids(20)
    print("\n全部stream_id序列：")
    print(ids)
```

## 题目三：在 NPU 上试用 3 个 torch\.accelerator 接口

**参考文件**（只读，用于追踪调用链，不改源码）：

- Python 入口：`pytorch_native/torch/accelerator/init.py`

- C\+\+ 绑定：`pytorch_native/torch/csrc/DeviceAccelerator.cpp`

- ATen 抽象：`pytorch_native/aten/src/ATen/DeviceAccelerator.cpp`（`at::getAccelerator`）

- 路由层：`pytorch_native/c10/core/impl/VirtualGuardImpl.h`

- NPU 后端实现：`pytorch_npu/torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp`

**任务**：

1. 编写一个 Python 脚本，`import torch` 后调用以下任意三个接口，在NPU上运行并打印返回值： [https://docs\.pytorch\.org/docs/2\.12/accelerator\.html](https://link.gitcode.com/?target=https%3A%2F%2Fdocs.pytorch.org%2Fdocs%2F2.12%2Faccelerator.html&from=https%3A%2F%2Fgitcode.com%2Fffmh%2Fpytorch%2Fwiki%2F%25E8%25AE%25BE%25E5%25A4%2587%25E7%25AE%25A1%25E7%2590%2586%25E8%25AF%25BE%25E7%25A8%258B.md&lang=zh&theme=white)

**验证**：日志

### 代码文件`test_accelerator_interface.py`

```Python
import torch

def test_three_accelerator_api():
    print("========== torch.accelerator 三接口测试（NPU后端） ==========\n")

    # 接口1：判断加速器是否可用
    avail = torch.accelerator.is_available()
    print(f"1. torch.accelerator.is_available() = {avail}")

    if avail:
        # 接口2：获取NPU设备总数
        dev_total = torch.accelerator.device_count()
        print(f"2. torch.accelerator.device_count() = {dev_total}")

        # 切换设备（消除弃用警告）
        torch.accelerator.set_device_index(0)

        # 接口3：获取当前设备编号（标准新接口，无警告）
        cur_dev_idx = torch.accelerator.current_device_index()
        print(f"3. torch.accelerator.current_device_index() = {cur_dev_idx}")

        # 张量校验改用原生npu写法，避开不存在的accelerator.device
        t = torch.rand(2, 2, device=f"npu:{cur_dev_idx}")
        print(f"\n张量校验：tensor.device = {t.device}")
    else:
        print("当前无可用NPU加速器，请检查torch_npu、驱动、固件环境")

if __name__ == "__main__":
    test_three_accelerator_api()
```


