import json

input_file = "data\RealToxicityPrompts\prompts.jsonl"
output_file = "data\RealToxicityPrompts\merged.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        try:
            data = json.loads(line)
            prompt_text = data.get("prompt", {}).get("text", "")
            continuation_text = data.get("continuation", {}).get("text", "")
            merged_text = prompt_text + continuation_text
            f_out.write(merged_text.replace('\n', ' ').strip() + "\n")
        except json.JSONDecodeError:
            continue  # 跳过无法解析的行

print(f"已完成，拼接结果保存为 {output_file}")
