import json
from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_bailian20231229.client import Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


# 初始化客户端
def create_client() -> Client:
    config = open_api_models.Config(
        access_key_id="LTAI5tAcRX7KyXHAr28Y8S6Y",
        access_key_secret="tIlyQhZwgxIhXOSxBACgTQYVG5POqG"
    )
    config.endpoint = "bailian.cn-beijing.aliyuncs.com"
    return Client(config)


# 读取JSONL文件
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]


# 主函数
def main():
    client = create_client()
    requests = read_jsonl("../../../finetune/dataset/output.jsonl")  # 你的JSONL请求文件

    # 构建批量推理请求体
    batch_request = bailian_models.BatchInferenceRequest(
        model_name="qwen2.5-coder-7B",  # 模型名称
        input=json.dumps(requests, ensure_ascii=False),  # 所有请求打包成字符串
        output_file_name="result_output.jsonl"  # 输出文件名
    )

    runtime = util_models.RuntimeOptions()
    try:
        response = client.batch_inference_with_options(batch_request, runtime)
        print("调用成功，响应：", response.body)
    except Exception as error:
        print("调用失败，错误信息：", error.message)


if __name__ == "__main__":
    main()