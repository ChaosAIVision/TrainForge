# 🚀 TrainForge - Framework Huấn Luyện AI Dễ Dàng

<div align="center">

![TrainForge Logo](https://img.shields.io/badge/TrainForge-AI%20Training%20Framework-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PHBhdGggZD0iTTEyIDJMNiA3VjE3TDEyIDIyTDE4IDE3VjdMMTIgMloiIHN0cm9rZT0iIzMwODRGRiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)

**Framework chuyên nghiệp để huấn luyện mô hình AI (LLM & VLM) một cách dễ dàng**

[Tính năng](#-tính-năng) • [Bắt đầu nhanh](#-bắt-đầu-nhanh) • [Cài đặt](#-cài-đặt) • [Sử dụng](#-sử-dụng) • [Ví dụ](#-ví-dụ) • [Cấu hình](#-cấu-hình)

</div>

---

## 🎯 TrainForge là gì?

TrainForge là một framework Python giúp bạn huấn luyện các mô hình AI một cách đơn giản và hiệu quả. Bạn chỉ cần:

1. **Tạo file cấu hình YAML** - Không cần code phức tạp
2. **Chạy 3 dòng lệnh** - Framework tự động xử lý mọi thứ
3. **Theo dõi kết quả** - Tích hợp sẵn Weights & Biases

**Phù hợp cho:** Sinh viên, nhà nghiên cứu, kỹ sư AI muốn huấn luyện mô hình nhanh chóng.

---

## ✨ Tính năng

| 🎯 **Đa nền tảng** | Hỗ trợ cả **Unsloth** (nhanh) và **HuggingFace** (ổn định) |
|---|---|
| 🤖 **Đa mô hình** | Huấn luyện cả **LLM** (mô hình ngôn ngữ) và **VLM** (mô hình thị giác-ngôn ngữ) |
| ⚡ **Tối ưu hóa** | Tự động tối ưu bộ nhớ và tốc độ huấn luyện |
| 🔧 **Cấu hình đơn giản** | Chỉ cần file YAML, không cần code phức tạp |
| 🚀 **Đa GPU** | Tự động phân phối huấn luyện trên nhiều GPU |
| 📊 **Theo dõi thí nghiệm** | Tích hợp sẵn Weights & Biases |

---

## 🚀 Bắt đầu nhanh

### Bước 1: Cài đặt
```bash
# Kích hoạt môi trường conda (nếu dùng unsloth)
conda activate unsloth

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Tạo file cấu hình
Tạo file `my_config.yaml`:

```yaml
model:
  model_name_or_path: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 4096
  load_in_4bit: true

hyperparams:
  per_device_train_batch_size: 1
  learning_rate: 0.0001
  num_train_epochs: 3
  output_dir: "my_model_output"

dataset:
  dataset_name: "ChaosAiVision/medical_1k_json"
  test_size: 0.1
  text_field: "text"
```

### Bước 3: Chạy huấn luyện
```python
from forge import UnslothLLMTrainer

# Khởi tạo trainer
trainer = UnslothLLMTrainer(config="my_config.yaml")

# Bắt đầu huấn luyện
trainer.train()
```

**Hoặc chạy trực tiếp:**
```bash
conda activate unsloth
PYTHONPATH=$PYTHONPATH:./src python examples/unsloth/llm/example_sft.py
```

---

## 📦 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- GPU NVIDIA (khuyến nghị)
- CUDA 11.8+ hoặc 12.0+

### Cài đặt từ source
```bash
# Clone repository
git clone https://github.com/yourusername/TrainForge.git
cd TrainForge

# Cài đặt dependencies
pip install -r requirements.txt

# Kiểm tra cài đặt
python -c "from src.forge import UnslothLLMTrainer; print('✅ Cài đặt thành công!')"
```

### Cài đặt môi trường Unsloth (khuyến nghị)
```bash
# Tạo môi trường conda mới
conda create --name unsloth python=3.11 -y
conda activate unsloth

# Cài đặt unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 💻 Sử dụng

### 1. Huấn luyện LLM (Mô hình ngôn ngữ)

```python
from forge import UnslothLLMTrainer

# Sử dụng file cấu hình có sẵn
trainer = UnslothLLMTrainer(config="config/unsloth/llm/sft.yaml")
results = trainer.train()

print(f"Huấn luyện hoàn thành! Kết quả: {results}")
```

### 2. Huấn luyện VLM (Mô hình thị giác-ngôn ngữ)

```python
from forge import UnslothVLMTrainer

# Huấn luyện mô hình có thể "nhìn" và "nói"
trainer = UnslothVLMTrainer(config="config/unsloth/vlm/sft.yaml")
results = trainer.train()
```

### 3. Huấn luyện đa GPU

```bash
# Sử dụng 2 GPU
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi-gpu --num_processes 2 train.py
```

---

## 📚 Ví dụ

### Ví dụ 1: Huấn luyện chatbot y tế
```python
"""
Huấn luyện mô hình trả lời câu hỏi y tế
"""
from forge import UnslothLLMTrainer

def main():
    print("🏥 Đang huấn luyện chatbot y tế...")
    
    # Sử dụng dataset y tế có sẵn
    trainer = UnslothLLMTrainer(config="config/unsloth/llm/sft.yaml")
    results = trainer.train()
    
    print("✅ Huấn luyện hoàn thành!")
    print(f"📊 Kết quả: {results}")

if __name__ == "__main__":
    main()
```

### Ví dụ 2: Huấn luyện mô hình hiểu ảnh
```python
"""
Huấn luyện mô hình có thể mô tả ảnh
"""
from forge import UnslothVLMTrainer

def main():
    print("👁️ Đang huấn luyện mô hình thị giác...")
    
    trainer = UnslothVLMTrainer(config="config/unsloth/vlm/sft.yaml")
    results = trainer.train()
    
    print("✅ Mô hình đã học cách 'nhìn' ảnh!")

if __name__ == "__main__":
    main()
```

---

## ⚙️ Cấu hình

### Cấu trúc file cấu hình

Mỗi file YAML có 5 phần chính:

```yaml
# 1. Cấu hình mô hình
model:
  model_name_or_path: "unsloth/Llama-3.2-3B-Instruct"  # Tên mô hình
  max_seq_length: 4096                                   # Độ dài tối đa
  load_in_4bit: true                                     # Tiết kiệm bộ nhớ

# 2. Tham số huấn luyện
hyperparams:
  per_device_train_batch_size: 1    # Kích thước batch
  learning_rate: 0.0001             # Tốc độ học
  num_train_epochs: 3               # Số epoch
  output_dir: "my_output"           # Thư mục lưu kết quả

# 3. Cấu hình layer
layer:
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  full_finetuning: false            # Chỉ huấn luyện một phần

# 4. Cấu hình dataset
dataset:
  dataset_name: "ChaosAiVision/medical_1k_json"  # Tên dataset
  test_size: 0.1                                  # Tỷ lệ test
  text_field: "text"                              # Trường chứa text

# 5. Cấu hình LoRA (tùy chọn)
lora:
  r: 8                              # Rank của LoRA
  lora_alpha: 32                    # Alpha parameter
  lora_dropout: 0.1                 # Dropout rate
```

### Các file cấu hình có sẵn

| File | Mô tả | Loại mô hình |
|------|-------|--------------|
| `config/unsloth/llm/sft.yaml` | Huấn luyện LLM với Unsloth | LLM |
| `config/unsloth/vlm/sft.yaml` | Huấn luyện VLM với Unsloth | VLM |

### Tham số quan trọng

#### Tối ưu bộ nhớ
- `load_in_4bit: true` - Giảm 75% bộ nhớ
- `per_device_train_batch_size: 1` - Batch size nhỏ
- `gradient_accumulation_steps: 4` - Tích lũy gradient

#### Tối ưu tốc độ
- `packing: true` - Đóng gói sequences (nhanh hơn 5x)
- `bf16: true` - Sử dụng bfloat16
- `dataloader_num_workers: 2` - Đa luồng load data

---

## 🛠️ Cấu trúc dự án

```
TrainForge/
├── src/forge/              # Mã nguồn chính
│   ├── core/              # Cấu hình cốt lõi
│   ├── module/            # Các backend (unsloth, huggingface)
│   ├── utils/             # Tiện ích
│   └── trainers.py        # Factory pattern
├── config/                # File cấu hình
│   ├── unsloth/          # Cấu hình Unsloth
│   └── huggingface/      # Cấu hình HuggingFace
├── examples/              # Ví dụ sử dụng
├── requirements.txt       # Dependencies
└── train.py              # Script huấn luyện chính
```

---

## 🚨 Xử lý lỗi thường gặp

### Lỗi: "No module named 'forge'"
```bash
# Giải pháp: Thêm src vào PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./src
# Hoặc
PYTHONPATH=$PYTHONPATH:./src python your_script.py
```

### Lỗi: "CUDA out of memory"
```yaml
# Giải pháp: Giảm batch size trong config
hyperparams:
  per_device_train_batch_size: 1  # Giảm từ 2 xuống 1
  gradient_accumulation_steps: 8  # Tăng để bù batch size
```

### Lỗi: "SFTTrainer unexpected keyword argument"
```python
# Đã được sửa trong phiên bản mới
# Nếu vẫn gặp lỗi, hãy cập nhật dependencies:
pip install --upgrade transformers trl
```

### Lỗi: Dataset không tìm thấy
```yaml
# Thay đổi dataset trong config
dataset:
  dataset_name: "your-dataset-name"  # Thay bằng dataset có sẵn
  # Hoặc sử dụng dataset local
  dataset_path: "path/to/your/data.json"
```

---

## 📊 Hiệu suất

### Benchmark trên các GPU phổ biến

| Mô hình | GPU | Thời gian huấn luyện | Bộ nhớ sử dụng |
|---------|-----|---------------------|----------------|
| Llama-3.2-3B | RTX 3090 (24GB) | 2h 30m | 12GB |
| Llama-3.2-3B | RTX 4090 (24GB) | 1h 45m | 10GB |
| Qwen2-VL-7B | 2x RTX 4090 | 6h 15m | 40GB |

### Mẹo tối ưu

1. **Sử dụng 4-bit quantization** - Tiết kiệm 75% bộ nhớ
2. **Bật packing** - Nhanh hơn 5x với sequences ngắn
3. **Điều chỉnh batch size** - Tùy theo bộ nhớ GPU
4. **Sử dụng đa GPU** - Cho mô hình lớn

---

## 🤝 Hỗ trợ

### Cần giúp đỡ?
- 🐛 **Báo lỗi**: [GitHub Issues](https://github.com/yourusername/TrainForge/issues)
- 💬 **Thảo luận**: [GitHub Discussions](https://github.com/yourusername/TrainForge/discussions)
- 📧 **Email**: support@trainforge.dev

### Câu hỏi thường gặp

**Q: Tôi có GPU 8GB, có thể huấn luyện được không?**
A: Có! Sử dụng `load_in_4bit: true` và `per_device_train_batch_size: 1`

**Q: Làm sao để huấn luyện với dataset của tôi?**
A: Thay đổi `dataset_name` trong file config hoặc sử dụng `dataset_path`

**Q: Tại sao huấn luyện chậm?**
A: Thử bật `packing: true` và tăng `per_device_train_batch_size`

---

## 📄 Giấy phép

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 🙏 Lời cảm ơn

- [Unsloth](https://github.com/unslothai/unsloth) - Tối ưu hóa huấn luyện
- [HuggingFace](https://huggingface.co/) - Transformers và datasets
- [Weights & Biases](https://wandb.ai/) - Theo dõi thí nghiệm

---

<div align="center">

**Được tạo với ❤️ bởi TrainForge Team**

[⭐ Star repo này](https://github.com/yourusername/TrainForge) • [🐛 Báo lỗi](https://github.com/yourusername/TrainForge/issues) • [📖 Tài liệu](https://docs.trainforge.dev)

**Chúc bạn huấn luyện AI thành công! 🚀**

</div>