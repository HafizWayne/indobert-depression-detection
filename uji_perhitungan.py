import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings

# Menonaktifkan peringatan yang tidak relevan dari transformers
warnings.filterwarnings("ignore", category=UserWarning)


# ----- FUNGSI HELPER (Diambil dari kode Streamlit) -----

def softmax(x):
    """Menghitung softmax untuk mengubah logits menjadi probabilitas."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def gelu(x):
    """Implementasi numerik dari fungsi aktivasi GELU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, epsilon=1e-6):
    """Melakukan normalisasi layer pada sebuah matriks."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + epsilon) + beta


# ----- FUNGSI INTI MODEL (Disederhanakan untuk klasifikasi) -----

def load_model_and_tokenizer(path):
    """Memuat model dan tokenizer dari path direktori."""
    try:
        if not os.path.isdir(path):
            print(f"❌ Error: Direktori tidak ditemukan di '{path}'.")
            return None, None
        print(f"🔄 Memuat model dan tokenizer dari '{path}'...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()  # Set model ke mode evaluasi
        print("✅ Model dan tokenizer berhasil dimuat.")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Gagal memuat model. Error: {e}")
        return None, None


def get_all_weights(model):
    """Mengekstrak semua bobot yang diperlukan dari model."""

    # Bobot Embedding
    embedding_weights = {
        "word_embeddings": model.bert.embeddings.word_embeddings.weight.data.numpy(),
        "position_embeddings": model.bert.embeddings.position_embeddings.weight.data.numpy(),
        "token_type_embeddings": model.bert.embeddings.token_type_embeddings.weight.data.numpy(),
        "ln_gamma": model.bert.embeddings.LayerNorm.weight.data.numpy(),
        "ln_beta": model.bert.embeddings.LayerNorm.bias.data.numpy()
    }

    # Bobot Layer Transformer
    encoder_weights = []
    for layer in model.bert.encoder.layer:
        layer_weights = {
            "q_w": layer.attention.self.query.weight.data.numpy().T,
            "k_w": layer.attention.self.key.weight.data.numpy().T,
            "v_w": layer.attention.self.value.weight.data.numpy().T,
            "attention_output_w": layer.attention.output.dense.weight.data.numpy().T,
            "attention_output_b": layer.attention.output.dense.bias.data.numpy(),
            "ln1_gamma": layer.attention.output.LayerNorm.weight.data.numpy(),
            "ln1_beta": layer.attention.output.LayerNorm.bias.data.numpy(),
            "ffn_w1": layer.intermediate.dense.weight.data.numpy().T,
            "ffn_b1": layer.intermediate.dense.bias.data.numpy(),
            "ffn_w2": layer.output.dense.weight.data.numpy().T,
            "ffn_b2": layer.output.dense.bias.data.numpy(),
            "ln2_gamma": layer.output.LayerNorm.weight.data.numpy(),
            "ln2_beta": layer.output.LayerNorm.bias.data.numpy(),
        }
        encoder_weights.append(layer_weights)

    # Bobot Classifier
    classifier_weights = {
        "W": model.classifier.weight.data.numpy().T,
        "b": model.classifier.bias.data.numpy()
    }

    return embedding_weights, encoder_weights, classifier_weights


def run_manual_prediction(token_ids, token_type_ids, model_config, all_weights):
    """Menjalankan forward pass manual untuk mendapatkan prediksi."""
    embedding_weights, encoder_weights, classifier_weights = all_weights

    # --- Tahap 1: Embedding ---
    word_embedding_vectors = embedding_weights["word_embeddings"][token_ids]
    position_ids = np.arange(len(token_ids))
    position_embedding_vectors = embedding_weights["position_embeddings"][position_ids]
    segment_embedding_vectors = embedding_weights["token_type_embeddings"][token_type_ids]

    summed_embeddings = word_embedding_vectors + position_embedding_vectors + segment_embedding_vectors
    hidden_states = layer_norm(summed_embeddings, embedding_weights['ln_gamma'], embedding_weights['ln_beta'])

    # --- Tahap 2: Transformer Encoder Layers ---
    num_heads = model_config.num_attention_heads
    head_size = model_config.hidden_size // num_heads

    for layer_weights in encoder_weights:
        current_input = hidden_states
        seq_len = current_input.shape[0]

        # Multi-Head Attention
        Q_proj = current_input @ layer_weights['q_w']
        K_proj = current_input @ layer_weights['k_w']
        V_proj = current_input @ layer_weights['v_w']

        Q_multi_head = Q_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)
        K_multi_head = K_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)
        V_multi_head = V_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)

        attention_scores = Q_multi_head @ K_multi_head.transpose(0, 2, 1)
        scaled_attention_scores = attention_scores / np.sqrt(head_size)
        attention_weights = softmax(scaled_attention_scores)
        attention_output_multi_head = attention_weights @ V_multi_head
        attention_output_concat = attention_output_multi_head.transpose(1, 0, 2).reshape(seq_len,
                                                                                         model_config.hidden_size)
        attention_output_final = attention_output_concat @ layer_weights['attention_output_w'] + layer_weights[
            'attention_output_b']

        # Add & Norm 1
        sum_1 = current_input + attention_output_final
        add_norm_1_output = layer_norm(sum_1, layer_weights['ln1_gamma'], layer_weights['ln1_beta'])

        # Feed-Forward Network
        ffn_intermediate = add_norm_1_output @ layer_weights['ffn_w1'] + layer_weights['ffn_b1']
        ffn_activated = gelu(ffn_intermediate)
        ffn_output_raw = ffn_activated @ layer_weights['ffn_w2'] + layer_weights['ffn_b2']

        # Add & Norm 2
        sum_2 = add_norm_1_output + ffn_output_raw
        hidden_states = layer_norm(sum_2, layer_weights['ln2_gamma'], layer_weights['ln2_beta'])

    # --- Tahap 3: Klasifikasi ---
    final_cls_output = hidden_states[0]  # Ambil output dari token [CLS]
    logits = final_cls_output @ classifier_weights['W'] + classifier_weights['b']
    probabilities = softmax(logits)

    return probabilities, logits


# ----- BLOK EKSEKUSI UTAMA -----

if __name__ == "__main__":
    # --- PENGATURAN ---
    MODEL_PATH = "finetuned_indobert_negation_combine"

    kalimat_untuk_diuji = [
        "aku merasa tidak_senang lagi",
        "hidupku terasa sangat tidak_berharga",
        "aku sudah tak_sanggup menjalani ini",
        "sepertinya tidak_ada harapan tersisa",
        "aku merasa ga_berguna untuk keluarga",
        "hatiku tidak_senang tanpa alasan jelas",
        "aku lelah merasa tidak_berharga",
        "aku tak_sanggup berpura pura kuat",
        "rasanya tidak_ada harapan untuk berubah",
        "aku ini memang ga_berguna",
        "aku tidak_senang saat sendirian",
        "kenapa aku selalu tidak_berharga",
        "aku tak_sanggup menahan semua ini",
        "sudah tidak_ada harapan bagiku",
        "aku ga_berguna dan menyusahkan",
        "aku tidak_senang dengan keadaanku",
        "menjadi tidak_berharga itu menyakitkan",
        "aku tak_sanggup menghadapi esok hari",
        "aku yakin tidak_ada harapan lagi",
        "aku lelah dianggap ga_berguna"
    ]

    # --- PROSES ---
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    if tokenizer and model:
        # Ekstrak bobot satu kali saja untuk efisiensi
        all_model_weights = get_all_weights(model)
        labels = ["Tidak Terindikasi", "Terindikasi"]

        print("\n" + "=" * 50)
        print("🚀 MEMULAI PROSES KLASIFIKASI 🚀")
        print("=" * 50 + "\n")

        for i, kalimat in enumerate(kalimat_untuk_diuji):
            # 1. Tokenisasi
            inputs = tokenizer(kalimat, return_tensors="pt", truncation=True, max_length=512)
            token_ids = inputs['input_ids'][0].tolist()
            token_type_ids = inputs['token_type_ids'][0].tolist()

            # 2. Lakukan Prediksi Manual
            probabilities, _ = run_manual_prediction(token_ids, token_type_ids, model.config, all_model_weights)

            # 3. Dapatkan Hasil
            prediction_idx = np.argmax(probabilities)
            prediction_label = labels[prediction_idx]

            # 4. Tampilkan Hasil
            print(f"--- Kalimat #{i + 1} ---")
            print(f"💬 Teks: \"{kalimat}\"")
            print(f"✅ Prediksi: {prediction_label}")

            # Tampilkan probabilitas untuk setiap label
            prob_text = " | ".join([f"{label}: {prob:.2%}" for label, prob in zip(labels, probabilities)])
            print(f"📊 Probabilitas: [ {prob_text} ]\n")