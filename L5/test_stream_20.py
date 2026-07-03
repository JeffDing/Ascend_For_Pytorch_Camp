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