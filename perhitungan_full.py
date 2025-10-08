import tempfile

import streamlit as st
import sys
import os



# GANTI FUNGSI PDF LAMA DENGAN VERSI LENGKAP INI


# TAMBAHKAN DUA BARIS INI DI ATAS UNTUK MEMBERSIHKAN CACHE
st.cache_data.clear()
st.cache_resource.clear()

def resource_path(relative_path):
    """ Mendapatkan path absolut ke resource, berfungsi untuk dev dan PyInstaller """
    try:
        # PyInstaller membuat folder temp _MEIPASS saat .exe berjalan
        base_path = sys._MEIPASS
    except Exception:
        # Saat berjalan sebagai script .py biasa, gunakan path normal
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


import pandas as pd
import numpy as np





# ----- FUNGSI HELPER -----

# Letakkan fungsi ini di bagian FUNGSI HELPER
def visualize_attention_as_text(tokens, scores):
    """Menghasilkan HTML untuk menyorot teks berdasarkan skor atensi."""
    # Normalisasi skor agar nilainya antara 0 dan 1 untuk intensitas warna
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    html_parts = []
    for token, score in zip(tokens, normalized_scores):
        # HSL: Hue(0=red), Saturation(100%), Lightness(100% - score*50%)
        # Semakin tinggi skor, semakin gelap warnanya (lightness menurun)
        color = f"hsl(0, 100%, {100 - score * 50}%)"
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 3px;">{token.replace("##", "")}</span>')

    return " ".join(html_parts)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Tambahkan fungsi GELU ini di bagian FUNGSI HELPER
def gelu(x):
    """Implementasi numerik dari fungsi aktivasi GELU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, epsilon=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + epsilon) + beta


def numpy_to_latex(arr, precision=2):
    if len(arr.shape) == 1: arr = arr.reshape(1, -1)
    rows = [" & ".join(map(lambda x: f"{x:.{precision}f}", row)) for row in arr]
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def tokens_to_latex_labels(tokens):
    cleaned_tokens = [token.replace('_', r'\_') for token in tokens]
    formatted_tokens = [f"\\text{{{token}}}" for token in cleaned_tokens]
    return r"\begin{matrix}" + r" \\ ".join(formatted_tokens) + r"\end{matrix}"


# ----- FUNGSI INTI MODEL -----

@st.cache_resource
def load_model_and_tokenizer(path=resource_path("finetuned_indobert_negation_combine")):
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    try:
        if not os.path.isdir(path):
            return None, None, f"Direktori tidak ditemukan di path: '{path}'. Pastikan path sudah benar."
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        # model.half()  # Menggunakan presisi float16 untuk hemat memori
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


# GANTI FUNGSI LAMA ANDA DENGAN VERSI LENGKAP INI
@st.cache_data
def load_negation_tokens(filepath=resource_path("token_negasi.txt")):
    """Membaca file teks dan memuat token negasi ke dalam sebuah set."""
    try:
        with open(filepath, "r") as f:
            # Menggunakan set untuk pencarian yang efisien (O(1) average time complexity)
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        st.error(f"File token negasi '{filepath}' tidak ditemukan. Pastikan file berada di folder yang sama.")
        return set()

def run_full_forward_pass(_model_path, token_ids):
    import torch
    # Muat model untuk mendapatkan konfigurasi dan bobot
    _, model, _ = load_model_and_tokenizer(_model_path)
    all_weights = get_model_weights(model)

    # Dapatkan konfigurasi penting dari model
    config = model.config
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads

    # --- PERUBAHAN 1: Ambil bobot dari layer embedding ---
    embedding_weights = {
        "word_embeddings": model.bert.embeddings.word_embeddings.weight.data.numpy(),
        "position_embeddings": model.bert.embeddings.position_embeddings.weight.data.numpy(),
    "token_type_embeddings": model.bert.embeddings.token_type_embeddings.weight.data.numpy(),
        "ln_gamma": model.bert.embeddings.LayerNorm.weight.data.numpy(),
        "ln_beta": model.bert.embeddings.LayerNorm.bias.data.numpy()
    }

    with torch.no_grad():
        initial_embeddings = model.bert.embeddings(torch.tensor(token_ids).unsqueeze(0)).numpy()[0]

    hidden_states = initial_embeddings
    all_hidden_states = [initial_embeddings]
    all_attention_weights = []
    all_inspection_data = []

    for i, layer_weights in enumerate(all_weights):
        current_input = hidden_states
        seq_len = current_input.shape[0]

        # --- Logika Multi-Head Attention ---

        # 1. Proyeksi Linear Awal
        Q_proj = current_input @ layer_weights['q_w']
        K_proj = current_input @ layer_weights['k_w']
        V_proj = current_input @ layer_weights['v_w']

        # 2. Reshape untuk Multi-Head
        Q_multi_head = Q_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)
        K_multi_head = K_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)
        V_multi_head = V_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)

        # 3. Perhitungan Skor Atensi Paralel
        attention_scores = Q_multi_head @ K_multi_head.transpose(0, 2, 1)
        scaled_attention_scores_multi_head = attention_scores / np.sqrt(head_size)
        attention_weights_multi_head = softmax(scaled_attention_scores_multi_head)

        # Simpan bobot atensi rata-rata dari semua kepala untuk visualisasi
        all_attention_weights.append(attention_weights_multi_head.mean(axis=0))

        # 4. Hitung Output Atensi Paralel
        attention_output_multi_head = attention_weights_multi_head @ V_multi_head

        # 5. Gabungkan (Concatenate) dan Reshape
        attention_output_concat = attention_output_multi_head.transpose(1, 0, 2).reshape(seq_len, config.hidden_size)

        # 6. Proyeksi Linear Akhir (W_o)
        attention_output_final = attention_output_concat @ layer_weights['attention_output_w'] + layer_weights[
            'attention_output_b']

        # --- Add & Norm, FFN ---

        sum_1 = current_input + attention_output_final
        add_norm_1_output = layer_norm(sum_1, layer_weights['ln1_gamma'], layer_weights['ln1_beta'])

        ffn_intermediate = add_norm_1_output @ layer_weights['ffn_w1'] + layer_weights['ffn_b1']
        ffn_activated = gelu(ffn_intermediate)
        ffn_output_raw = ffn_activated @ layer_weights['ffn_w2'] + layer_weights['ffn_b2']

        sum_2 = add_norm_1_output + ffn_output_raw
        hidden_states = layer_norm(sum_2, layer_weights['ln2_gamma'], layer_weights['ln2_beta'])
        all_hidden_states.append(hidden_states)

        # --- PENGISIAN ULANG INSPECTION_DATA SECARA LENGKAP ---
        current_layer_inspection_data = {
            "input": current_input,
            "Q": Q_proj,  # Menggunakan matriks sebelum di-reshape
            "K": K_proj,  # Menggunakan matriks sebelum di-reshape
            "V": V_proj,  # Menggunakan matriks sebelum di-reshape
            # Untuk UI, kita tampilkan rata-rata dari semua kepala
            "scaled_attention_scores": scaled_attention_scores_multi_head,
            "attention_weights": attention_weights_multi_head,
            # Variabel ini adalah hasil gabungan sebelum proyeksi linear akhir
            "attention_output": attention_output_concat,
            "attention_output_final": attention_output_final,
            "sum_1": sum_1,
            "add_norm_1_output": add_norm_1_output,
            "ffn_intermediate": ffn_intermediate,
            "ffn_activated": ffn_activated,  # Menggantikan 'ffn_relu'
            "ffn_output": ffn_output_raw,
            "sum_2": sum_2,
            "final_output": hidden_states,
            "head_size": head_size
        }
        all_inspection_data.append(current_layer_inspection_data)

    final_cls_output = hidden_states[0]
    return final_cls_output, all_inspection_data, all_weights, all_hidden_states, all_attention_weights, embedding_weights


def get_model_weights(_model):
    weights = []
    for layer in _model.bert.encoder.layer:
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
        weights.append(layer_weights)
    return weights


def get_classifier_weights(_model):
    # Mengambil bobot dari layer classifier
    return {
        "W": _model.classifier.weight.data.numpy().T,
        "b": _model.classifier.bias.data.numpy()
    }


# ----- UI APLIKASI STREAMLIT -----

st.set_page_config(layout="wide")
st.title("🔬 Analisis Atensi & Perhitungan Detail Transformer")
st.header("1. Pengaturan Model dan Input")
model_path = st.text_input("Masukkan Path Direktori Model Anda:", resource_path("finetuned_indobert_negation_combine"))
kalimat = st.text_input("Masukkan Kalimat Studi Kasus:",
                        "aku lelah merasa tidak_berharga")
# Membuat daftar pilihan untuk dropdown
layer_options = list(range(1, 13))

# Menggunakan st.selectbox sebagai pengganti st.slider
detail_layer = st.selectbox(
    "Pilih Layer untuk Analisis Detail:",
    options=layer_options,
    index=11  # Atur default ke Layer 12 (karena 12 ada di indeks ke-11)
)
detail_layer_idx = detail_layer - 1  # Baris ini tetap sama dan masih diperlukan
show_details = st.checkbox("Tampilkan Contoh Perhitungan Spesifik (memperlambat aplikasi)")

if st.button("🚀 Proses dan Analisis Kalimat"):
    from fpdf import FPDF
    import matplotlib.pyplot as plt


    class PDF(FPDF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_counter = 0
            # Mendaftarkan font proporsional (untuk teks biasa)
            self.add_font("DejaVu", "", resource_path("DejaVuSans.ttf"))
            self.add_font("DejaVu", "B", resource_path("DejaVuSans-Bold.ttf"))
            self.add_font("DejaVu", "I", resource_path("DejaVuSans-Oblique.ttf"))
            self.font_family_regular = "DejaVu"

            # Mendaftarkan font monospaced (untuk tabel/data)
            self.add_font("DejaVuMono", "", resource_path("DejaVuSansMono.ttf"))
            self.add_font("DejaVuMono", "B", resource_path("DejaVuSansMono-Bold.ttf"))
            self.font_family_mono = "DejaVuMono"

        def header(self):
            self.set_font(self.font_family_regular, 'B', 12)
            self.cell(0, 10, 'Rincian Lengkap Perhitungan Matematis', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_font(self.font_family_regular, 'I', 8)
            self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

        def section_title(self, title):
            self.set_font(self.font_family_regular, 'B', 12)
            self.cell(0, 6, title, 0, 1, 'L')
            self.ln(2)

        def subsection_title(self, title):
            self.set_font(self.font_family_regular, 'B', 10)
            self.cell(0, 5, title, 0, 1, 'L')
            self.ln(1)

        def body_text(self, text):
            self.set_font(self.font_family_regular, '', 10)
            self.multi_cell(0, 5, text)
            self.ln(2)

        def write_latex(self, formula, font_size=12):
            # ... (fungsi ini sudah benar dan tidak perlu diubah) ...
            fig, ax = plt.subplots(figsize=(6, 1), dpi=200)
            ax.text(0.5, 0.5, f'${formula}$', size=font_size, ha='center', va='center')
            ax.axis('off')

            temp_filename = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    plt.savefig(tmpfile, format='png', bbox_inches='tight', pad_inches=0.1)
                    temp_filename = tmpfile.name
                plt.close(fig)
                self.image(temp_filename, x=self.get_x() + 10, w=self.w - 40)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            self.ln(2)

        def write_dataframe(self, df, tokens=None, limit=4):
            """Menulis DataFrame, memotong kolom jika terlalu lebar."""
            self.set_font(self.font_family_mono, '', 8)

            matrix = df.to_numpy()
            num_rows, num_cols = matrix.shape

            # Jika matriks tidak terlalu lebar, tampilkan seperti biasa
            if num_cols <= (2 * limit) + 1:
                df_text = df.to_string()
                self.multi_cell(0, 4, df_text)
            # Jika terlalu lebar, potong dan tampilkan ringkasannya
            else:
                lines = []
                for i in range(num_rows):
                    token_label = tokens[i] if tokens and i < len(tokens) else str(df.index[i])
                    token_prefix = f"{token_label:<15}"

                    start_slice = matrix[i, :limit]
                    end_slice = matrix[i, -limit:]

                    start_str = " ".join([f"{val:6.2f}" for val in start_slice])
                    end_str = " ".join([f"{val:6.2f}" for val in end_slice])

                    full_row_str = f"{start_str}  ...  {end_str}"
                    lines.append(token_prefix + full_row_str)

                final_text = "\n".join(lines)
                self.multi_cell(0, 4, final_text)

            self.ln(2)


    def generate_full_math_pdf(
            kalimat, tokens, token_ids, inputs, embedding_weights, all_weights,
            all_inspection_data, detail_layer, detail_layer_idx, show_details,
            model_config, classifier_weights, logits, probabilities, labels
    ):
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        selected_layer_data = all_inspection_data[detail_layer_idx]

        # --- BAGIAN 1: EMBEDDING (Sesuai Streamlit) ---
        pdf.section_title("Langkah 1: Teks Mentah ke Final Embedding (Input H⁰)")
        pdf.body_text("Proses 'penerjemahan' dari teks menjadi matriks angka sebelum diproses oleh Layer 1.")

        pdf.subsection_title("1a. Tokenisasi & Konversi ke ID")
        token_df = pd.DataFrame(
            {'Token': tokens, 'Token ID': token_ids, 'Segment ID': inputs['token_type_ids'][0].tolist()})
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4, token_df.to_string())
        pdf.ln(2)

        word_embedding_vectors = embedding_weights["word_embeddings"][token_ids]
        position_ids = np.arange(len(token_ids))
        position_embedding_vectors = embedding_weights["position_embeddings"][position_ids]
        token_type_ids = inputs['token_type_ids'][0].numpy()
        segment_embedding_vectors = embedding_weights["token_type_embeddings"][token_type_ids]
        summed_embeddings = word_embedding_vectors + position_embedding_vectors + segment_embedding_vectors
        final_embeddings_h0 = layer_norm(summed_embeddings, embedding_weights['ln_gamma'], embedding_weights['ln_beta'])

        pdf.subsection_title("1b. Word Embedding (E_word)")
        pdf.write_dataframe(pd.DataFrame(word_embedding_vectors), tokens=tokens)

        pdf.subsection_title("1c. Position Embedding (E_pos)")
        pdf.write_dataframe(pd.DataFrame(position_embedding_vectors), tokens=tokens)

        pdf.subsection_title("1d. Segment Embedding (E_seg)")
        pdf.write_dataframe(pd.DataFrame(segment_embedding_vectors), tokens=tokens)

        pdf.subsection_title("1e. Penjumlahan, Normalisasi, dan Final Embedding (H⁰)")
        pdf.write_latex(r"H^{(0)} = \text{LayerNorm}(E_{word} + E_{pos} + E_{seg})")
        pdf.body_text("Hasil Matriks Embedding Final H^(0):")
        pdf.write_dataframe(pd.DataFrame(final_embeddings_h0), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk H^(0)[0,0]")
            e_word_00 = word_embedding_vectors[0, 0]
            e_pos_00 = position_embedding_vectors[0, 0]
            e_seg_00 = segment_embedding_vectors[0, 0]
            sum_00 = summed_embeddings[0, 0]
            pdf.write_latex(
                rf"\text{{1. Penjumlahan: }} {e_word_00:.2f} + {e_pos_00:.2f} + {e_seg_00:.2f} = {sum_00:.2f}")

            sum_row_0 = summed_embeddings[0]
            gamma_vector = embedding_weights['ln_gamma']
            beta_vector = embedding_weights['ln_beta']
            h0_00 = final_embeddings_h0[0, 0]
            epsilon = 1e-6
            mean_val = np.mean(sum_row_0)
            std_val = np.std(sum_row_0)
            pdf.body_text(f"2. Normalisasi (Mean μ={mean_val:.4f}, Std Dev σ={std_val:.4f}):")
            formula = rf"H^{{(0)}}[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - ({mean_val:.4f})}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + ({beta_vector[0]:.2f}) = {h0_00:.2f}"
            pdf.write_latex(formula)

        # --- BAGIAN 2: TRANSFORMER LAYER (Sesuai Streamlit) ---
        pdf.add_page()
        pdf.section_title(f"Perhitungan Detail untuk Layer {detail_layer}")

        pdf.subsection_title("Langkah 2: Input H dan Proyeksi Q, K, V")
        pdf.write_latex(r"Q = H \cdot W_Q \quad | \quad K = H \cdot W_K \quad | \quad V = H \cdot W_V")
        pdf.body_text(
            "Matriks input dari layer sebelumnya (H) diproyeksikan menjadi tiga matriks: Query (Q), Key (K), dan Value (V).")

        pdf.body_text(f"Matriks Input H^({detail_layer - 1}):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['input']), tokens=tokens)

        # --- [KODE BARU DITAMBAHKAN DI SINI] ---
        pdf.body_text("Hasil Matriks Q (Query):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['Q']), tokens=tokens)

        pdf.body_text("Hasil Matriks K (Key):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['K']), tokens=tokens)

        pdf.body_text("Hasil Matriks V (Value):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['V']), tokens=tokens)
        # --- [AKHIR DARI KODE BARU] ---

        pdf.subsection_title("Matriks Bobot (Weights) untuk Q, K, V (slice 5x5)")
        wq_slice = pd.DataFrame(all_weights[detail_layer_idx]['q_w'][:5, :5])
        wk_slice = pd.DataFrame(all_weights[detail_layer_idx]['k_w'][:5, :5])
        wv_slice = pd.DataFrame(all_weights[detail_layer_idx]['v_w'][:5, :5])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W_q (slice):\n{wq_slice.to_string(float_format='%.2f')}\n\nW_k (slice):\n{wk_slice.to_string(float_format='%.2f')}\n\nW_v (slice):\n{wv_slice.to_string(float_format='%.2f')}")
        pdf.ln(2)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk Q, K, V [0,0]")
            h_row0 = selected_layer_data['input'][0]
            wq_col0 = all_weights[detail_layer_idx]['q_w'][:, 0]
            q_val00 = selected_layer_data['Q'][0, 0]
            q_terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_row0, wq_col0)]
            pdf.write_latex(rf"Q[0,0] = {q_terms[0]} + \dots = {q_val00:.2f}")

            wk_col0 = all_weights[detail_layer_idx]['k_w'][:, 0]
            k_val00 = selected_layer_data['K'][0, 0]
            k_terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_row0, wk_col0)]
            pdf.write_latex(rf"K[0,0] = {k_terms[0]} + \dots = {k_val00:.2f}")

            wv_col0 = all_weights[detail_layer_idx]['v_w'][:, 0]
            v_val00 = selected_layer_data['V'][0, 0]
            v_terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_row0, wv_col0)]
            pdf.write_latex(rf"V[0,0] = {v_terms[0]} + \dots = {v_val00:.2f}")

        pdf.subsection_title("Langkah 3: Perhitungan Skor Atensi")
        pdf.write_latex(r"\text{Scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}")

        # Tentukan kepala atensi yang akan ditampilkan (konsisten dengan contoh)
        head_idx_to_show = 0

        pdf.body_text(f"Hasil Matriks Skor Atensi (dari Kepala ke-{head_idx_to_show + 1}, sebelum Softmax):")
        scores_one_head = selected_layer_data['scaled_attention_scores'][head_idx_to_show]
        pdf.write_dataframe(pd.DataFrame(scores_one_head), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk Scores[0,0]")
            head_idx_to_show = 0
            q_vector_0 = selected_layer_data['Q'][0]
            k_vector_0 = selected_layer_data['K'][0]
            raw_score_00 = np.dot(q_vector_0, k_vector_0)
            head_size = model_config.hidden_size // model_config.num_attention_heads
            scaled_score_00 = selected_layer_data['scaled_attention_scores'][head_idx_to_show, 0, 0]

            # --- [PERUBAHAN UTAMA] ---
            # Buat beberapa suku pertama dari dot product untuk ditampilkan
            dot_product_terms = [f"({q:.2f} \cdot {k:.2f})" for q, k in zip(q_vector_0[:3], k_vector_0[:3])]

            pdf.body_text("1. Dot Product (perkalian antara baris Q[0] dan K[0]):")
            pdf.write_latex(
                rf"\text{{Skor Mentah}} = {dot_product_terms[0]} + {dot_product_terms[1]} + {dot_product_terms[2]} + \dots = {raw_score_00:.2f}")

            pdf.body_text("2. Scaling (hasil dot product dibagi dengan akar dari d_k):")
            pdf.write_latex(
                rf"\text{{Skor Akhir}} = \frac{{{raw_score_00:.2f}}}{{\sqrt{{{head_size}}}}} = {scaled_score_00:.2f}")

        pdf.subsection_title("Langkah 4: Normalisasi Skor dengan Softmax")
        pdf.write_latex(r"W_{att} = \text{softmax}(\text{Scores})")

        # Gunakan indeks kepala atensi yang sama seperti di Langkah 3
        head_idx_to_show = 0

        pdf.body_text(f"Hasil Matriks Bobot Atensi (W_att) (dari Kepala ke-{head_idx_to_show + 1}):")
        weights_one_head = selected_layer_data['attention_weights'][head_idx_to_show]
        pdf.write_dataframe(pd.DataFrame(weights_one_head), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk W_att[0,0] (dari Kepala ke-1)")
            head_idx_to_show = 0
            scores_row0 = selected_layer_data['scaled_attention_scores'][head_idx_to_show, 0]
            stable_scores = scores_row0 - np.max(scores_row0)
            exp_s0 = np.exp(stable_scores[0])
            sum_exp = np.sum(np.exp(stable_scores))
            w_att00 = selected_layer_data['attention_weights'][head_idx_to_show, 0, 0]

            # --- [PERUBAHAN UTAMA] ---
            # Buat beberapa suku pertama dari penyebut (denominator) softmax
            denominator_terms = [f"\exp({s:.2f})" for s in scores_row0[:3]]

            formula = rf"W_{{att}}[0,0] = \frac{{\exp({scores_row0[0]:.2f})}}{{{denominator_terms[0]} + {denominator_terms[1]} + {denominator_terms[2]} + \dots}} = \frac{{{exp_s0:.2f}}}{{{sum_exp:.2f}}} = {w_att00:.4f}"
            pdf.write_latex(formula)

        pdf.subsection_title("Langkah 5a: Penggabungan Kontekstual")
        pdf.write_latex(r"\text{AttentionOutput} = W_{att} \cdot V")
        pdf.body_text("Hasil Matriks AttentionOutput (sebelum proyeksi):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['attention_output']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk AttentionOutput[0,0] (dari Kepala ke-1)")
            head_idx_to_show = 0
            watt_row0 = selected_layer_data['attention_weights'][head_idx_to_show, 0]
            v_col0 = selected_layer_data['V'][:, 0]
            # Note: att_out_val00 is from the concatenated output, so this is an illustrative example of one head's contribution.
            att_out_val00 = selected_layer_data['attention_output'][0, 0]
            terms = [f"({w:.2f} \cdot {v:.2f})" for w, v in zip(watt_row0, v_col0)]
            pdf.write_latex(rf"\text{{AttOut}}[0,0] = {terms[0]} + \dots = {att_out_val00:.2f} \text{{ (ilustrasi)}}")

        pdf.subsection_title("Langkah 5b: Proyeksi Linear Akhir Atensi")
        pdf.write_latex(r"\text{AttentionOutputFinal} = \text{AttentionOutput} \cdot W_o + b_o")
        pdf.subsection_title("Matriks Bobot (W_o & b_o) (slice)")
        wo_matrix = all_weights[detail_layer_idx]['attention_output_w']
        bo_vector = all_weights[detail_layer_idx]['attention_output_b']
        wo_slice = pd.DataFrame(wo_matrix[:5, :5])
        bo_slice = pd.DataFrame(bo_vector[:5], columns=['bias'])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W_o (slice 5x5):\n{wo_slice.to_string(float_format='%.2f')}\n\nb_o (slice awal):\n{bo_slice.to_string(float_format='%.2f')}")
        pdf.ln(2)
        pdf.body_text("Hasil Matriks AttentionOutputFinal:")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['attention_output_final']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk AttOutFinal[0,0]")
            att_out_row0 = selected_layer_data['attention_output'][0]
            wo_col0 = all_weights[detail_layer_idx]['attention_output_w'][:, 0]
            bo_0 = all_weights[detail_layer_idx]['attention_output_b'][0]
            att_out_final_val00 = selected_layer_data['attention_output_final'][0, 0]
            terms = [f"({att:.2f} \cdot {w:.2f})" for att, w in zip(att_out_row0, wo_col0)]
            pdf.write_latex(
                rf"\text{{AttOutFinal}}[0,0] = ({terms[0]} + \dots) + {bo_0:.2f} = {att_out_final_val00:.2f}")

        pdf.subsection_title("Langkah 6: Add & Norm (Setelah Atensi)")
        pdf.write_latex(r"H' = \text{LayerNorm}(H_{in} + \text{AttentionOutputFinal})")
        pdf.body_text("Hasil Matriks H' (Input untuk FFN):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['add_norm_1_output']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk H'[0,0]")
            h_in_00 = selected_layer_data['input'][0, 0]
            att_out_00 = selected_layer_data['attention_output_final'][0, 0]
            sum_00 = selected_layer_data['sum_1'][0, 0]
            pdf.write_latex(rf"\text{{1. Penjumlahan: }} {h_in_00:.2f} + {att_out_00:.2f} = {sum_00:.2f}")
            sum_row_0 = selected_layer_data['sum_1'][0]
            gamma_vector = all_weights[detail_layer_idx]['ln1_gamma']
            beta_vector = all_weights[detail_layer_idx]['ln1_beta']
            h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
            epsilon = 1e-6
            mean_val = np.mean(sum_row_0)
            std_val = np.std(sum_row_0)
            pdf.body_text(f"2. Normalisasi (Mean μ={mean_val:.4f}, Std Dev σ={std_val:.4f}):")
            formula = rf"H'[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - ({mean_val:.4f})}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + ({beta_vector[0]:.2f}) = {h_prime_00:.2f}"
            pdf.write_latex(formula)

        pdf.section_title("Langkah 7: Feed-Forward Network (FFN)")
        pdf.write_latex(r"\text{FFN}(H') = \text{GELU}(H'W_1 + b_1)W_2 + b_2")
        pdf.subsection_title("Matriks Bobot untuk FFN (slice)")
        w1_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_w1'][:5, :5])
        b1_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_b1'][:5], columns=['bias'])
        w2_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_w2'][:5, :5])
        b2_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_b2'][:5], columns=['bias'])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W1 (slice 5x5):\n{w1_slice.to_string(float_format='%.2f')}\n\n"
                       f"b1 (slice awal):\n{b1_slice.to_string(float_format='%.2f')}\n\n"
                       f"W2 (slice 5x5):\n{w2_slice.to_string(float_format='%.2f')}\n\n"
                       f"b2 (slice awal):\n{b2_slice.to_string(float_format='%.2f')}"
                       )
        pdf.ln(2)

        pdf.subsection_title("7a. Hasil setelah Linear Pertama")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['ffn_intermediate']), tokens=tokens)
        pdf.subsection_title("7b. Hasil setelah Aktivasi GELU")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['ffn_activated']), tokens=tokens)
        pdf.subsection_title("7c. Hasil setelah Linear Kedua")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['ffn_output']), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail FFN")
            h_prime_row0 = selected_layer_data['add_norm_1_output'][0]
            w1_col0 = all_weights[detail_layer_idx]['ffn_w1'][:, 0]
            b1_0 = all_weights[detail_layer_idx]['ffn_b1'][0]
            ffn_inter_00 = selected_layer_data['ffn_intermediate'][0, 0]
            terms1 = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_prime_row0, w1_col0)]
            pdf.write_latex(rf"\text{{1. Linear 1: }} {terms1[0]} + \dots + {b1_0:.2f} = {ffn_inter_00:.2f}")

            pdf.write_latex(
                rf"\text{{2. Aktivasi GELU: GELU}}({ffn_inter_00:.2f}) \to {selected_layer_data['ffn_activated'][0, 0]:.2f}")

            ffn_activated_row0 = selected_layer_data['ffn_activated'][0]
            w2_col0 = all_weights[detail_layer_idx]['ffn_w2'][:, 0]
            b2_0 = all_weights[detail_layer_idx]['ffn_b2'][0]
            ffn_output_00 = selected_layer_data['ffn_output'][0, 0]
            terms2 = [f"({r:.2f} \cdot {w:.2f})" for r, w in zip(ffn_activated_row0, w2_col0)]
            pdf.write_latex(rf"\text{{3. Linear 2: }} {terms2[0]} + \dots + {b2_0:.2f} = {ffn_output_00:.2f}")

        pdf.subsection_title("Langkah 8: Add & Norm Final")
        pdf.write_latex(r"H_{out} = \text{LayerNorm}(H' + \text{FFN}(H'))")
        pdf.body_text("Hasil Matriks Final H_out:")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['final_output']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk H_out[0,0]")
            h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
            ffn_out_00 = selected_layer_data['ffn_output'][0, 0]
            sum_2_00 = selected_layer_data['sum_2'][0, 0]
            pdf.write_latex(rf"\text{{1. Penjumlahan: }} {h_prime_00:.2f} + {ffn_out_00:.2f} = {sum_2_00:.2f}")
            sum_row_2 = selected_layer_data['sum_2'][0]
            gamma_vector_2 = all_weights[detail_layer_idx]['ln2_gamma']
            beta_vector_2 = all_weights[detail_layer_idx]['ln2_beta']
            h_out_00 = selected_layer_data['final_output'][0, 0]
            mean_val_2 = np.mean(sum_row_2)
            std_val_2 = np.std(sum_row_2)
            pdf.body_text(f"2. Normalisasi (Mean μ={mean_val_2:.4f}, Std Dev σ={std_val_2:.4f}):")
            formula_2 = rf"H_{{out}}[0,0] = {gamma_vector_2[0]:.2f} \times \frac{{{sum_2_00:.2f} - ({mean_val_2:.4f})}}{{\sqrt{{{std_val_2 ** 2:.4f} + \epsilon}}}} + ({beta_vector_2[0]:.2f}) = {h_out_00:.2f}"
            pdf.write_latex(formula_2)

        # --- BAGIAN 3: KLASIFIKASI FINAL ---
        pdf.add_page()
        pdf.section_title("Tahap Klasifikasi Final")
        prediction = labels[np.argmax(probabilities)]

        pdf.subsection_title("Langkah 9: Perhitungan Logits")
        pdf.write_latex(r"Z = h_{CLS, final} \cdot W_{classifier} + b_{classifier}")
        pdf.body_text("Bobot Klasifikasi:")
        w_cls_slice = pd.DataFrame(classifier_weights['W'][:5, :])
        b_cls = pd.DataFrame(pd.Series(classifier_weights['b']), columns=['bias'])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W_classifier (slice 5x{b_cls.shape[0]}):\n{w_cls_slice.to_string(float_format='%.2f')}\n\nb_classifier:\n{b_cls.to_string(float_format='%.2f')}")
        pdf.ln(2)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk Logit Z[0]")
            final_cls_output = selected_layer_data['final_output'][0]
            terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(final_cls_output, classifier_weights['W'][:, 0])]
            pdf.write_latex(rf"Z[0] = ({terms[0]} + \dots) + {classifier_weights['b'][0]:.2f} = {logits[0]:.2f}")

        pdf.subsection_title("Langkah 10: Probabilitas Akhir")
        pdf.write_latex(r"P_k = \frac{e^{Z_k}}{\sum_{j} e^{Z_j}}")
        pdf.body_text("Hasil Probabilitas:")
        prob_str = ""
        for label, prob in zip(labels, probabilities):
            prob_str += f"- {label}: {prob:.2%}\n"
        pdf.body_text(prob_str)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk Probabilitas")
            exp_z0 = np.exp(logits[0])
            sum_exp_final = np.sum(np.exp(logits))
            prob0 = probabilities[0]
            formula = rf"P(\text{{{labels[0]}}}) = \frac{{\exp({logits[0]:.2f})}}{{\dots}} = \frac{{{exp_z0:.2f}}}{{{sum_exp_final:.2f}}} = {prob0:.4f}"
            pdf.write_latex(formula)

        pdf.subsection_title("Keputusan Akhir")
        pdf.set_font(pdf.font_family_regular, 'B', 12)
        pdf.cell(0, 10, f'Prediksi Model: {prediction}', 0, 1, 'L')

        return bytes(pdf.output(dest='S'))

    tokenizer, model, error = load_model_and_tokenizer(model_path)
    if error:
        st.error(f"Gagal memuat model. **Pesan Error:** {error}")
    else:
        st.success(f"Model dari direktori '{model_path}' berhasil dimuat.")
        inputs = tokenizer(kalimat, return_tensors="pt")
        token_ids = inputs['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        st.markdown("**Hasil Tokenisasi:**");
        st.info(f"`{tokens}`")

        # --- [BLOK BARU] PENGENALAN POLA NEGASI ---
        st.markdown("---")
        st.subheader("✔️ Pengenalan Pola Negasi Khusus")

        # Muat daftar token negasi dari file Anda
        negation_tokens_set = load_negation_tokens()

        # Cari token dari kalimat input yang ada di dalam daftar Anda
        found_patterns = [token for token in tokens if token in negation_tokens_set]

        if found_patterns:
            # Jika ada token yang ditemukan, tampilkan
            display_text = ", ".join(f"`{pattern}`" for pattern in found_patterns)
            st.success(f"Pola negasi khusus terdeteksi di dalam kalimat: {display_text}")
        else:
            # Jika tidak ada, beri tahu pengguna
            st.info("Tidak ada pola negasi khusus dari daftar Anda yang terdeteksi di dalam kalimat ini.")
        # --- [AKHIR BLOK BARU] ---

        with st.spinner("Mensimulasikan forward pass dan mengumpulkan data dari semua layer..."):
            final_cls_output, all_inspection_data, all_weights, all_hidden_states, all_attention_weights, embedding_weights = run_full_forward_pass(
                model_path, tuple(token_ids))
            classifier_weights = get_classifier_weights(model)
            logits = final_cls_output @ classifier_weights['W'] + classifier_weights['b']
            probabilities = softmax(logits)
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            prediction = labels[np.argmax(probabilities)]

            # --- DEFINISIKAN FILTER DI SINI (SEBELUM HEADER 2) ---
            tokens_to_ignore = {'[cls]', '[sep]', '[pad]', '.', ',', '?', '!', ':', ';'}
            # --- BLOK KODE YANG DIPERBARUI UNTUK HEADER 2 ---
            st.header("2. Rangkuman Fokus Atensi [CLS] di Setiap Layer")
            st.markdown(
                "Bagian ini menunjukkan token **kata** mana (selain `[CLS]` dan tanda baca) yang menjadi fokus utama di setiap layer.")

            # --- PENJELASAN FASE DIMASUKKAN DI SINI ---

            st.subheader("Fase 1: Analisis Sintaksis & Konteks Dasar (Layer 1-4)")
            st.markdown(
                "Model fokus pada hubungan gramatikal dan makna dasar kata berdasarkan kata di sekitarnya.")

            st.subheader("Fase 2: Pembangunan Makna Semantik (Layer 5-8)")
            st.markdown(
                "Model mulai merangkai makna yang lebih dalam dan mengidentifikasi konsep emosional atau niat dalam kalimat.")
            st.subheader("Fase 3: Kontekstualisasi & Kesimpulan (Layer 9-12)")
            st.markdown(
                "Model mengunci sinyal paling penentu dan mengumpulkan informasi")

            for i, attention_matrix in enumerate(all_attention_weights):
                # Buat daftar token dan skor yang sudah difilter untuk layer ini
                filtered_layer_tokens = []
                filtered_layer_scores = []
                # Gunakan 'tokens' asli dan 'attention_matrix' dari layer saat ini
                for token, score in zip(tokens, attention_matrix[0]):
                    if token.lower() not in tokens_to_ignore:
                        clean_token = token.replace('##', '')
                        filtered_layer_tokens.append(clean_token)
                        filtered_layer_scores.append(score)

                # Cari token dengan skor tertinggi HANYA dari daftar yang sudah difilter
                if filtered_layer_tokens:  # Pastikan daftar tidak kosong setelah difilter
                    max_index = np.argmax(filtered_layer_scores)
                    most_attended_token = filtered_layer_tokens[max_index]
                    max_score = filtered_layer_scores[max_index]
                    st.markdown(
                        f"- **Layer {i + 1}**: Fokus utama pada token **`{most_attended_token}`** (skor: {max_score:.2f})")
            # --- AKHIR DARI BLOK YANG DIPERBARUI ---

            st.header(f"3. Analisis Atensi Detail (Layer {detail_layer})")
            cls_attention_selected_layer = all_attention_weights[detail_layer_idx][0]

            # Buat daftar baru yang sudah bersih untuk grafik (logika yang sama seperti di atas)
            filtered_tokens = []
            filtered_scores = []

            # Loop melalui token dan skor asli
            for token, score in zip(tokens, cls_attention_selected_layer):
                if token.lower() not in tokens_to_ignore:
                    clean_token = token.replace('##', '')
                    filtered_tokens.append(clean_token)
                    filtered_scores.append(score)

            # Buat grafik menggunakan data yang sudah difilter
            fig, ax = plt.subplots(figsize=(10, 4));

            ax.bar(filtered_tokens, filtered_scores, color='skyblue')

            ax.set_ylabel("Skor Atensi");
            ax.set_title(f"Bobot Atensi dari [CLS] ke Token Relevan (Layer {detail_layer})")
            plt.xticks(rotation=45, ha="right");
            plt.tight_layout()
            st.pyplot(fig)

            # Cara memanggilnya di UI Anda (di Header 3, setelah membuat grafik bar)
            st.subheader("Visualisasi Atensi dalam Kalimat")
            html_output = visualize_attention_as_text(np.array(filtered_tokens), np.array(filtered_scores))
            st.markdown(html_output, unsafe_allow_html=True)


            # GANTI SELURUH BLOK EXPANDER ANDA DENGAN INI
            st.header(f"5. Rincian Perhitungan Matematis (Layer {detail_layer})")

            # --- BLOK TOMBOL UNDUH PDF LENGKAP ---
            st.markdown("Unduh **seluruh rincian perhitungan matematis** yang ditampilkan di bawah dalam format PDF.")

            # Panggil fungsi generator PDF yang baru dan lengkap
            # Panggil fungsi generator PDF dengan argumen LENGKAP
            pdf_data = generate_full_math_pdf(
                kalimat=kalimat,
                tokens=tokens,
                token_ids=token_ids,
                inputs=inputs,
                embedding_weights=embedding_weights,
                all_weights=all_weights,
                all_inspection_data=all_inspection_data,
                detail_layer=detail_layer,
                detail_layer_idx=detail_layer_idx,
                model_config=model.config,
                show_details=show_details,
                # --- TAMBAHKAN 4 ARGUMEN YANG HILANG INI ---
                classifier_weights=classifier_weights,
                logits=logits,
                probabilities=probabilities,
                labels=labels
            )

            st.download_button(
                label="📥 Unduh Perhitungan Lengkap (PDF)",
                data=pdf_data,
                file_name=f"perhitungan_lengkap_layer_{detail_layer}.pdf",
                mime="application/pdf"
            )
            # --- AKHIR BLOK TOMBOL UNDUH ---

            with st.expander(
                    f"Klik untuk melihat semua rumus, matriks, dan contoh perhitungan untuk Layer {detail_layer}"):
                # Ambil data inspeksi untuk layer yang dipilih dari slider (sudah benar)
                selected_layer_data = all_inspection_data[detail_layer_idx]

                # GANTI BAGIAN AWAL DI DALAM EXPANDER ANDA DENGAN BLOK KODE LENGKAP INI

                # GANTI BAGIAN AWAL DI DALAM EXPANDER ANDA DENGAN BLOK KODE LENGKAP INI

                st.subheader("Langkah 1: Teks Mentah ke Final Embedding (Input H⁰)")
                st.markdown(
                    "Ini adalah proses 'penerjemahan' dari teks yang bisa dibaca manusia menjadi matriks angka yang dipahami model, sebelum diproses oleh Layer 1.")

                # --- 1a. Tokenisasi & Konversi ke ID ---
                st.markdown("---");
                st.markdown("#### 1a. Tokenisasi & Konversi ke ID")
                st.markdown("Kalimat dipecah menjadi token, dan setiap token dipetakan ke sebuah nomor ID unik.")
                token_df = pd.DataFrame(
                    {'Token': tokens, 'Token ID': token_ids, 'Segment ID': inputs['token_type_ids'][0].tolist()})
                st.dataframe(token_df)

                # --- 1b. Word Embeddings ---
                st.markdown("---");
                st.markdown("#### 1b. Pembentukan Word Embedding")
                st.markdown("Setiap Token ID digunakan untuk mengambil vektor maknanya dari matriks embedding.")
                word_embedding_vectors = embedding_weights["word_embeddings"][token_ids]
                st.write(f"**Vektor Word Embedding (E_word) untuk kalimat Anda:**")
                st.dataframe(word_embedding_vectors.round(2))

                # --- 1c. Position Embeddings ---
                st.markdown("---");
                st.markdown("#### 1c. Pembentukan Position Embedding")
                st.markdown("Vektor posisi ditambahkan untuk memberi informasi urutan kata pada model.")
                position_ids = np.arange(len(token_ids))
                position_embedding_vectors = embedding_weights["position_embeddings"][position_ids]
                st.write(f"**Vektor Position Embedding (E_pos) untuk kalimat Anda:**")
                st.dataframe(position_embedding_vectors.round(2))

                # --- 1d. Segment Embeddings ---
                st.markdown("---");
                st.markdown("#### 1d. Pembentukan Segment Embedding")
                st.markdown(
                    "Vektor segmen ditambahkan untuk membedakan antar kalimat (dalam kasus ini, semua adalah segmen 0).")
                token_type_ids = inputs['token_type_ids'][0].numpy()

                segment_embedding_vectors = embedding_weights["token_type_embeddings"][token_type_ids]
                st.write(f"**Vektor Segment Embedding (E_seg) untuk kalimat Anda:**")
                st.dataframe(segment_embedding_vectors.round(2))

                # --- 1e. Penjumlahan & Normalisasi ---
                st.markdown("---");
                st.markdown("#### 1e. Penjumlahan, Normalisasi, dan Final Embedding (H⁰)")
                st.markdown("Ketiga vektor embedding dijumlahkan, lalu distabilkan dengan Layer Normalization.")
                st.latex(r"H^{(0)} = \text{LayerNorm}(E_{word} + E_{pos} + E_{seg})")

                # Lakukan penjumlahan dan normalisasi manual
                summed_embeddings = word_embedding_vectors + position_embedding_vectors + segment_embedding_vectors
                final_embeddings_h0 = layer_norm(summed_embeddings, embedding_weights['ln_gamma'],
                                                 embedding_weights['ln_beta'])

                st.markdown("**Hasil Matriks Embedding Final `H^(0)` (Input untuk Layer 1):**")
                col_f1, col_f2 = st.columns([1, 10]);
                with col_f1:
                    st.latex(tokens_to_latex_labels(tokens))
                with col_f2:
                    st.latex(f"H^{{(0)}} = {numpy_to_latex(final_embeddings_h0, 2)}")

                # --- TAMBAHAN: Contoh Perhitungan Detail untuk Embedding ---
                if show_details:
                    st.markdown("✍️ **Contoh Perhitungan Detail untuk Elemen Pertama `H^(0)[0,0]`**:");

                    # 1. Contoh Penjumlahan
                    st.markdown("**1. Penjumlahan Embedding (element-wise):**")
                    e_word_00 = word_embedding_vectors[0, 0]
                    e_pos_00 = position_embedding_vectors[0, 0]
                    e_seg_00 = segment_embedding_vectors[0, 0]
                    sum_00 = summed_embeddings[0, 0]
                    st.latex(rf"Sum[0,0] = E_{{word}}[0,0] + E_{{pos}}[0,0] + E_{{seg}}[0,0]")
                    st.latex(rf"{sum_00:.2f} = {e_word_00:.2f} + {e_pos_00:.2f} + {e_seg_00:.2f}")

                    # 2. Contoh Layer Normalization
                    st.markdown("**2. Normalisasi (LayerNorm):**")
                    st.markdown(
                        "Fungsi `LayerNorm` diterapkan pada **seluruh baris pertama** dari matriks hasil penjumlahan di atas.")
                    sum_row_0 = summed_embeddings[0]
                    gamma_vector = embedding_weights['ln_gamma']
                    beta_vector = embedding_weights['ln_beta']
                    h0_00 = final_embeddings_h0[0, 0]
                    epsilon = 1e-6

                    mean_val = np.mean(sum_row_0)
                    std_val = np.std(sum_row_0)

                    st.markdown(f"   - **Mean (μ) dari seluruh elemen di baris pertama:** `{mean_val:.4f}`")
                    st.markdown(f"   - **Standar Deviasi (σ) dari seluruh elemen di baris ini:** `{std_val:.4f}`")

                    st.latex(r"H^{(0)}[i, j] = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j")

                    calc_str = rf"H^{{(0)}}[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - {mean_val:.4f}}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + {beta_vector[0]:.2f}"
                    calc_result = rf" = {h0_00:.2f}"
                    st.latex(calc_str + calc_result)

                # --- AKHIR DARI BLOK PENGGANTI ---

                st.subheader("Langkah 1 & 2: Input H dan Hasil Q, K, V")
                st.latex(r"Q = H \cdot W_Q \quad | \quad K = H \cdot W_K \quad | \quad V = H \cdot W_V")
                st.markdown(
                    f"**Matriks Input `H^{{({detail_layer - 1})}}` dan Hasil Matriks `Q`, `K`, `V` untuk Layer {detail_layer}:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Input H**");
                    st.dataframe(selected_layer_data['input'].round(2))
                with col2:
                    st.markdown("**Hasil Q**");
                    st.dataframe(selected_layer_data['Q'].round(2))
                with col3:
                    st.markdown("**Hasil K**");
                    st.dataframe(selected_layer_data['K'].round(2))
                with col4:
                    st.markdown("**Hasil V**");
                    st.dataframe(selected_layer_data['V'].round(2))

                st.markdown("**Matriks Bobot (Weights) dari Model (ditampilkan sebagian 5x5):**")
                # --- PERBAIKAN 1: Gunakan detail_layer_idx untuk mengambil bobot yang benar ---
                wq_matrix = all_weights[detail_layer_idx]['q_w']
                wk_matrix = all_weights[detail_layer_idx]['k_w']
                wv_matrix = all_weights[detail_layer_idx]['v_w']

                w_col1, w_col2, w_col3 = st.columns(3)
                with w_col1:
                    st.write("$W_Q$ (slice)");
                    st.dataframe(wq_matrix[:5, :5].round(2))
                with w_col2:
                    st.write("$W_K$ (slice)");
                    st.dataframe(wk_matrix[:5, :5].round(2))
                with w_col3:
                    st.write("$W_V$ (slice)");
                    st.dataframe(wv_matrix[:5, :5].round(2))

                # --- PERBAIKAN 2: Blok duplikat yang menampilkan H, Q, K, V lagi dihapus ---

                # --- PERBAIKAN 3: Gunakan selected_layer_data untuk semua perhitungan detail ---
                if show_details:
                    h_row0 = selected_layer_data['input'][0]
                    st.markdown("✍️ **Contoh Perhitungan Elemen Pertama (baris 0, kolom 0):**")

                    wq_col0 = all_weights[detail_layer_idx]['q_w'][:, 0]
                    q_val00 = selected_layer_data['Q'][0, 0]
                    terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_row0, wq_col0)]
                    st.latex(rf"Q[0,0] = {' + '.join(terms)} = {q_val00:.2f}")

                    wk_col0 = all_weights[detail_layer_idx]['k_w'][:, 0]
                    k_val00 = selected_layer_data['K'][0, 0]
                    terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_row0, wk_col0)]
                    st.latex(rf"K[0,0] = {' + '.join(terms)} = {k_val00:.2f}")

                    wv_col0 = all_weights[detail_layer_idx]['v_w'][:, 0]
                    v_val00 = selected_layer_data['V'][0, 0]
                    terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_row0, wv_col0)]
                    st.latex(rf"V[0,0] = {' + '.join(terms)} = {v_val00:.2f}")

                st.markdown("---");
                st.subheader("Langkah 4: Normalisasi Skor dengan Softmax")
                st.latex(r"W_{att} = \text{softmax}(\text{Attention Scores})")

                st.markdown("---");
                st.subheader(f"Matriks Bobot Atensi (W_att) untuk Layer {detail_layer}")
                st.markdown(
                    "Ini adalah matriks yang **benar-benar digunakan** oleh model untuk langkah selanjutnya. Model menggunakan semua matriks dari ke-12 kepala secara bersamaan.")

                # Ambil matriks atensi 3D lengkap (12, panjang_token, panjang_token)
                full_attention_matrix = selected_layer_data['attention_weights']
                num_heads = full_attention_matrix.shape[0]

                # Buat daftar nama untuk setiap tab
                tab_names = [f"Kepala {i + 1}" for i in range(num_heads)]

                # Buat container tab untuk menampilkan matriks setiap kepala
                tabs = st.tabs(tab_names)

                head_idx_for_example = 0  # Pilih Kepala 1 untuk dijadikan contoh perhitungan

                for i, tab in enumerate(tabs):
                    with tab:
                        head_matrix = full_attention_matrix[i]
                        df_head = pd.DataFrame(head_matrix, index=tokens, columns=tokens)
                        styler = df_head.style.background_gradient(cmap='viridis', axis=None).format("{:.2f}")

                        # Beri highlight jika ini adalah kepala yang digunakan untuk contoh
                        if i == head_idx_for_example:
                            st.info(
                                f"Baris pertama di-highlight karena digunakan pada contoh perhitungan detail di bawah (Kepala {i + 1}).")
                            styler = styler.set_properties(**{'background-color': '#fde047'},
                                                           subset=pd.IndexSlice[tokens[0], :])

                        st.dataframe(styler)

                # --- BAGIAN CONTOH PERHITUNGAN DETAIL (TETAP ADA JIKA DICENTANG) ---
                if show_details:
                    st.markdown("---")
                    st.subheader(f"Contoh Perhitungan Detail (Layer {detail_layer}, dari Kepala Atensi ke-1)")

                    # Tentukan kepala atensi mana yang akan digunakan sebagai contoh
                    head_idx_for_example = 0

                    # --- BAGIAN 1: PERHITUNGAN SKOR ATENSI ---
                    st.markdown("**Perhitungan Skor Atensi `Scores[0,0]`**")

                    # Ambil data yang diperlukan
                    q_vector_0 = selected_layer_data['Q'][0]
                    k_vector_0 = selected_layer_data['K'][0]
                    head_size = selected_layer_data['head_size']  # Diambil dari data inspeksi
                    scaled_score_00 = selected_layer_data['scaled_attention_scores'][head_idx_for_example, 0, 0]

                    # Hitung skor mentah
                    raw_score_00 = np.dot(q_vector_0, k_vector_0)

                    # Tampilkan perhitungan
                    st.latex(
                        rf"\text{{1. Dot Product: }} Q[0] \cdot K[0] = ({q_vector_0[0]:.2f} \cdot {k_vector_0[0]:.2f}) + \dots = {raw_score_00:.2f}")
                    st.latex(
                        rf"\text{{2. Scaling: }} \frac{{{raw_score_00:.2f}}}{{\sqrt{{{head_size}}}}} = {scaled_score_00:.2f}")

                    # --- BAGIAN 2: PERHITUNGAN BOBOT ATENSI ---
                    st.markdown("**Perhitungan Bobot Atensi `W_att[0,0]`**")

                    # Ambil baris skor dari kepala atensi yang dipilih
                    scores_row0 = selected_layer_data['scaled_attention_scores'][head_idx_for_example, 0]

                    # Terapkan "max trick" untuk stabilitas numerik
                    stable_scores = scores_row0 - np.max(scores_row0)

                    # Hitung nilai untuk contoh, menggunakan skor yang sudah stabil
                    exp_s0 = np.exp(stable_scores[0])
                    sum_exp = np.sum(np.exp(stable_scores))
                    w_att00 = selected_layer_data['attention_weights'][head_idx_for_example, 0, 0]

                    # Tampilkan formula dengan nilai yang sudah benar dan stabil
                    st.latex(
                        rf"W_{{att}}[0,0] = \frac{{\exp({scores_row0[0]:.2f})}}{{\exp({scores_row0[0]:.2f}) + \exp({scores_row0[1]:.2f}) + \dots}} = \frac{{{exp_s0:.2f}}}{{{sum_exp:.2f}}} = {w_att00:.4f}")
                st.markdown("---");
                st.subheader("Langkah 5: Menghitung Output Atensi")
                st.markdown("---");
                st.subheader("Langkah 5a: Menghitung Output Atensi Awal (AttOut)")
                st.latex(r"\text{AttentionOutput} = W_{att} \cdot V")
                st.markdown("**Hasil Matriks `AttentionOutput`:**")
                st.dataframe(selected_layer_data['attention_output'].round(2))

                if show_details:
                    if show_details:
                        st.markdown("✍️ **Contoh Perhitungan Elemen `Att_Out[0,0]` (dari Kepala Atensi ke-1)**:");

                        # 1. Pilih satu kepala atensi untuk dijadikan contoh (misal, indeks 0)
                        head_idx_to_show = 0

                        # 2. Ambil SATU BARIS dari kepala atensi yang dipilih (INI PERBAIKANNYA)
                        watt_row0 = selected_layer_data['attention_weights'][head_idx_to_show, 0]

                        # Sisa kode sudah benar
                        v_col0 = selected_layer_data['V'][:, 0]
                        att_out_val00 = selected_layer_data['attention_output'][0, 0]
                        terms = [rf"({w:.2f} \times {v:.2f})" for w, v in zip(watt_row0, v_col0)]
                        st.latex(rf"\text{{Att\_Out}}[0,0] = {' + '.join(terms)} = {att_out_val00:.2f}")

                st.markdown("---");
                st.subheader("Langkah 5b: Proyeksi Linear dari Output Atensi")
                st.markdown(
                    "Hasil atensi awal kemudian diproses oleh sebuah layer linear (Dense) untuk menghasilkan output final dari blok atensi.")
                st.latex(r"\text{AttentionOutputFinal} = \text{AttentionOutput} \cdot W_o + b_o")
                st.markdown("**Matriks Bobot untuk Proyeksi Linear (ditampilkan sebagian):**")
                wo_matrix = all_weights[detail_layer_idx]['attention_output_w']
                bo_vector = all_weights[detail_layer_idx]['attention_output_b']
                w_col1, w_col2 = st.columns(2)
                with w_col1:
                    st.write("$W_o$ (slice 5x5)")
                    st.dataframe(wo_matrix[:5, :5].round(2))
                with w_col2:
                    st.write("$b_o$ (slice awal)")
                    st.dataframe(bo_vector[:5].round(2))
                st.markdown("**Hasil Matriks `AttentionOutputFinal`:**")
                st.dataframe(selected_layer_data['attention_output_final'].round(2))

                if show_details:
                    st.markdown("✍️ **Contoh Perhitungan Elemen `AttOutFinal[0,0]`:**");
                    att_out_row0 = selected_layer_data['attention_output'][0]
                    wo_col0 = all_weights[detail_layer_idx]['attention_output_w'][:, 0]
                    bo_0 = all_weights[detail_layer_idx]['attention_output_b'][0]
                    att_out_final_val00 = selected_layer_data['attention_output_final'][0, 0]

                    terms = [rf"({att:.2f} \times {w:.2f})" for att, w in zip(att_out_row0, wo_col0)]
                    st.latex(rf"\text{{AttOutFinal}}[0,0] = ({' + '.join(terms)}) + {bo_0:.2f} = {att_out_final_val00:.2f}")

                st.markdown("---");
                # --- [BAGIAN YANG DIPERBARUI] Langkah 6: Add & Norm, lalu FFN ---
                st.markdown("---");
                st.subheader("Langkah 6: Add & Norm (Setelah Atensi)")
                st.latex(r"H' = \text{LayerNorm}(H_{in} + \text{AttentionOutput})")
                st.markdown("**Hasil Matriks `H'` (Input untuk FFN):**")
                st.dataframe(selected_layer_data['add_norm_1_output'].round(2))
                # Bagian 1: Penjumlahan (Add)
                h_in_00 = selected_layer_data['input'][0, 0]
                att_out_00 = selected_layer_data['attention_output_final'][0, 0]
                sum_00 = selected_layer_data['sum_1'][0, 0]
                st.latex(
                    rf"\text{{1. Penjumlahan: }} H_{{in}}[0,0] + \text{{AttOutFinal}}[0,0] = {h_in_00:.2f} + {att_out_00:.2f} = {sum_00:.2f}")

                # --- Bagian 2: Normalisasi (Norm) ---
                st.markdown(
                    "2. **Normalisasi**: Fungsi `LayerNorm` diterapkan pada **seluruh baris pertama** dari matriks hasil penjumlahan di atas.")

                # Ambil data yang diperlukan untuk perhitungan
                sum_row_0 = selected_layer_data['sum_1'][0]
                gamma_vector = all_weights[i]['ln1_gamma']
                beta_vector = all_weights[i]['ln1_beta']
                h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
                epsilon = 1e-6

                # Hitung mean dan std
                mean_val = np.mean(sum_row_0)
                std_val = np.std(sum_row_0)

                st.markdown(
                    f"   - **Vektor baris pertama (sebelum normalisasi) dimulai dengan:** `[{sum_row_0[0]:.2f}, {sum_row_0[1]:.2f}, {sum_row_0[2]:.2f}, ...]`")
                st.markdown(f"   - **Hitung Mean (μ) dari seluruh elemen di baris ini:** `{mean_val:.4f}`")
                st.markdown(f"   - **Hitung Standar Deviasi (σ) dari seluruh elemen di baris ini:** `{std_val:.4f}`")

                st.markdown("   - **Terapkan rumus LayerNorm untuk elemen pertama** ($H'[0,0]$):")
                st.latex(r"H'[i, j] = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j")

                calc_str = rf"H'[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - {mean_val:.4f}}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + {beta_vector[0]:.2f}"
                calc_result = rf" = {h_prime_00:.2f}"
                st.latex(calc_str + calc_result)

                st.markdown("---");
                st.markdown("---");
                st.subheader("Langkah 7: Feed-Forward Network (FFN)")
                st.latex(r"\text{FFN}(H') = \max(0, H'W_1 + b_1)W_2 + b_2")

                st.markdown("**Matriks Bobot untuk FFN (ditampilkan sebagian):**")
                w1_matrix = all_weights[i]['ffn_w1']
                b1_vector = all_weights[i]['ffn_b1']
                w2_matrix = all_weights[i]['ffn_w2']
                b2_vector = all_weights[i]['ffn_b2']
                ffn_w_col1, ffn_w_col2 = st.columns(2)
                with ffn_w_col1:
                    st.write("$W_1$ (slice 5x5)");
                    st.dataframe(w1_matrix[:5, :5].round(2))
                    st.write("$b_1$ (slice awal)");
                    st.dataframe(b1_vector[:5].round(2))
                with ffn_w_col2:
                    st.write("$W_2$ (slice 5x5)");
                    st.dataframe(w2_matrix[:5, :5].round(2))
                    st.write("$b_2$ (slice awal)");
                    st.dataframe(b2_vector[:5].round(2))

                st.markdown("**7a. Hasil setelah Linear Pertama ($H'W_1 + b_1$):**")
                st.dataframe(selected_layer_data['ffn_intermediate'].round(2))
                st.markdown("**7b. Hasil setelah Aktivasi ReLU (max(0, ...)):**")
                st.dataframe(selected_layer_data['ffn_activated'].round(2))
                st.markdown("**7c. Hasil setelah Linear Kedua (Output FFN):**")
                st.dataframe(selected_layer_data['ffn_output'].round(2))

                if show_details:
                    st.markdown("✍️ **Contoh Perhitungan Elemen Pertama di FFN (baris 0, kolom 0):**")
                    h_prime_row0 = selected_layer_data['add_norm_1_output'][0]
                    w1_col0 = all_weights[i]['ffn_w1'][:, 0]
                    b1_0 = all_weights[i]['ffn_b1'][0]
                    ffn_inter_00 = selected_layer_data['ffn_intermediate'][0, 0]
                    terms1 = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_prime_row0, w1_col0)]
                    st.latex(
                        rf"\text{{1. Linear Pertama: }} \text{{FFN\_inter}}[0,0] = ({' + '.join(terms1)}) + {b1_0:.2f} = {ffn_inter_00:.2f}")

                    # --- PERUBAHAN DI SINI ---
                    st.latex(
                        rf"\text{{2. Aktivasi GELU: }} \text{{GELU}}({ffn_inter_00:.2f}) \to {selected_layer_data['ffn_activated'][0, 0]:.2f}")

                    # --- DAN DI SINI ---
                    ffn_activated_row0 = selected_layer_data['ffn_activated'][0]  # Mengganti 'ffn_relu'
                    w2_col0 = all_weights[i]['ffn_w2'][:, 0]
                    b2_0 = all_weights[i]['ffn_b2'][0]
                    ffn_output_00 = selected_layer_data['ffn_output'][0, 0]

                    # --- DAN DI SINI ---
                    terms2 = [rf"({r:.2f} \times {w:.2f})" for r, w in
                              zip(ffn_activated_row0, w2_col0)]  # Menggunakan variabel baru
                    st.latex(
                        rf"\text{{3. Linear Kedua: }} \text{{FFN\_out}}[0,0] = ({' + '.join(terms2)}) + {b2_0:.2f} = {ffn_output_00:.2f}")

                    # --- [BAGIAN YANG DIPERBARUI] Langkah 8 ---
                    st.markdown("---");
                    st.subheader("Langkah 8: Add & Norm Final")
                    st.latex(r"H_{out} = \text{LayerNorm}(H' + \text{FFN}(H'))")
                    st.markdown("**Hasil Matriks Final `H_out` (Output dari Layer 1):**")
                    st.dataframe(selected_layer_data['final_output'].round(2))

                    if show_details:
                        st.markdown("✍️ **Contoh Perhitungan Elemen `H_out[0,0]`:**")

                        # Bagian 1: Penjumlahan
                        h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
                        ffn_out_00 = selected_layer_data['ffn_output'][0, 0]
                        sum_2_00 = selected_layer_data['sum_2'][0, 0]
                        st.latex(
                            rf"\text{{1. Penjumlahan: }} H'[0,0] + \text{{FFN}}[0,0] = {h_prime_00:.2f} + {ffn_out_00:.2f} = {sum_2_00:.2f}")

                        # Bagian 2: Normalisasi (Versi Lengkap)
                        st.markdown(
                            "2. **Normalisasi**: Fungsi `LayerNorm` diterapkan pada seluruh baris pertama dari hasil penjumlahan di atas.")

                        sum_row_2 = selected_layer_data['sum_2'][0]
                        gamma_vector_2 = all_weights[i]['ln2_gamma']
                        beta_vector_2 = all_weights[i]['ln2_beta']
                        h_out_00 = selected_layer_data['final_output'][0, 0]
                        epsilon = 1e-6

                        mean_val_2 = np.mean(sum_row_2)
                        std_val_2 = np.std(sum_row_2)

                        st.markdown(f"   - **Hitung Mean (μ) dari seluruh elemen di baris ini:** `{mean_val_2:.4f}`")
                        st.markdown(
                            f"   - **Hitung Standar Deviasi (σ) dari seluruh elemen di baris ini:** `{std_val_2:.4f}`")
                        st.markdown("   - **Terapkan rumus LayerNorm untuk elemen pertama** ($H_{out}[0,0]$):")
                        st.latex(r"H_{out}[i, j] = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j")

                        calc_str = rf"H_{{out}}[0,0] = {gamma_vector_2[0]:.2f} \times \frac{{{sum_2_00:.2f} - {mean_val_2:.4f}}}{{\sqrt{{{std_val_2 ** 2:.4f} + \epsilon}}}} + {beta_vector_2[0]:.2f}"
                        calc_result = rf" = {h_out_00:.2f}"
                        st.latex(calc_str + calc_result)

    # [BAGIAN YANG DIPERBARUI] Tahap Klasifikasi Final
    st.header("5. Tahap Klasifikasi Final")
    st.write("Vektor `[CLS]` dari output layer terakhir (Layer 12) digunakan untuk prediksi akhir.")
    st.latex(f"h_{{CLS, final}} = {numpy_to_latex(final_cls_output, 2)}")

    st.subheader("Langkah 9: Perhitungan Logits")
    st.latex(r"Z = h_{CLS, final} \cdot W_{classifier} + b_{classifier}")

    W_classifier = classifier_weights['W'];
    b_classifier = classifier_weights['b']
    st.markdown("**Bobot Klasifikasi (ditampilkan sebagian):**")
    col_w, col_b = st.columns(2)
    with col_w:
        st.write("$W_{classifier}$ (slice 5x2)");
        st.dataframe(W_classifier[:5, :].round(2))
    with col_b:
        st.write("$b_{classifier}$");
        st.dataframe(b_classifier.round(2))

    st.markdown("**Hasil Logits (Z):**")
    st.latex(f"Z = {numpy_to_latex(logits)}")

    if show_details:
        st.markdown("✍️ **Contoh Perhitungan Logit Pertama `Z[0]`:**")
        terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(final_cls_output, W_classifier[:, 0])]
        st.latex(rf"Z[0] = ({' + '.join(terms)}) + {b_classifier[0]:.2f} = {logits[0]:.2f}")

    st.subheader("Langkah 10: Fungsi Softmax & Probabilitas Akhir")
    st.latex(r"P_k = \frac{e^{Z_k}}{\sum_{j} e^{Z_j}}")

    labels = ["Tidak Terindikasi", "Terindikasi Depresi"]
    prediction = labels[np.argmax(probabilities)]

    st.markdown("**Hasil Probabilitas:**")
    for i, label in enumerate(labels):
        st.markdown(f"- **{label}**: {probabilities[i]:.2%}")

    if show_details:
        st.markdown("✍️ **Contoh Perhitungan Probabilitas:**")
        exp_z0 = np.exp(logits[0]);
        exp_z1 = np.exp(logits[1])
        sum_exp = exp_z0 + exp_z1
        st.latex(
            rf"P(\text{{{labels[0]}}}) = \frac{{e^{{{logits[0]:.2f}}}}}{{e^{{{logits[0]:.2f}}} + e^{{{logits[1]:.2f}}}}} = \frac{{{exp_z0:.2f}}}{{{sum_exp:.2f}}} = {probabilities[0]:.4f}")
        st.latex(
            rf"P(\text{{{labels[1]}}}) = \frac{{e^{{{logits[1]:.2f}}}}}{{e^{{{logits[0]:.2f}}} + e^{{{logits[1]:.2f}}}}} = \frac{{{exp_z1:.2f}}}{{{sum_exp:.2f}}} = {probabilities[1]:.4f}")

    st.subheader("Keputusan Akhir")
    st.success(f"## **Prediksi Model: {prediction}**")

    # --- TAMBAHKAN BLOK RINGKASAN NARATIF DI SINI ---
    st.markdown("---")
    st.header("📜 Ringkasan Cerita Analisis")

    # Kumpulkan token-token kunci dari rangkuman Header 2
    key_tokens_per_layer = []
    for i, attention_matrix in enumerate(all_attention_weights):
        filtered_layer_tokens = []
        filtered_layer_scores = []
        for token, score in zip(tokens, attention_matrix[0]):
            if token.lower() not in tokens_to_ignore:
                clean_token = token.replace('##', '')
                filtered_layer_tokens.append(clean_token)
                filtered_layer_scores.append(score)
        if filtered_layer_tokens:
            max_index = np.argmax(filtered_layer_scores)
            key_tokens_per_layer.append(filtered_layer_tokens[max_index])
        else:
            key_tokens_per_layer.append("")  # Tambahkan string kosong jika tidak ada token

    # Buat narasi menggunakan f-string
    summary_narrative = f"""
        Secara ringkas, begitulah perjalanan "pikiran" model saat menganalisis kalimat Anda:

        1.  **Fase Awal (Layer 1-4):** Model pertama-tama membedah struktur kalimat dan dengan cepat mengidentifikasi kata kunci yang menunjukkan kondisi, seperti **`{key_tokens_per_layer[1]}`** dan **`{key_tokens_per_layer[2]}`**.

        2.  **Fase Tengah (Layer 5-8):** Selanjutnya, ia mulai membangun pemahaman yang lebih dalam. Fokusnya mulai bergeser untuk memahami konteks yang lebih luas dari kondisi tersebut, seperti yang ditunjukkan oleh perhatian pada kata **`{key_tokens_per_layer[5]}`** dan **`{key_tokens_per_layer[7]}`**.

        3.  **Fase Akhir (Layer 9-12):** Pada tahap final, model mengunci niat atau sentimen paling kuat dari kalimat. Ini terlihat dari fokusnya yang konsisten pada kata-kata penentu seperti **`{key_tokens_per_layer[8]}`** dan **`{key_tokens_per_layer[10]}`**.

        Berdasarkan akumulasi pemahaman dari awal hingga akhir inilah, model akhirnya mengambil keputusan bahwa kalimat tersebut **{prediction}** dengan probabilitas **{probabilities[np.argmax(probabilities)]:.2%}**.
        """
    st.success(summary_narrative)
    # --- AKHIR DARI BLOK RINGKASAN ---

# --- GANTI BLOK LAMA ANDA DENGAN BLOK BARU INI ---

if __name__ == '__main__':
    import multiprocessing
    import os
    import sys

    # Baris ini penting untuk mencegah error rekursi proses
    multiprocessing.freeze_support()

    # --- BAGIAN PENTING DIMULAI DI SINI ---
    # Cek apakah "flag" sudah ada. Jika ya, berarti ini adalah eksekusi kedua, jadi berhenti.
    if os.environ.get("STREAMLIT_RUNNING_MAIN"):
        sys.exit()

    # Jika ini eksekusi pertama, atur "flag" sebelum memanggil Streamlit
    os.environ["STREAMLIT_RUNNING_MAIN"] = "true"
    # --- BAGIAN PENTING BERAKHIR DI SINI ---

    import streamlit.web.cli as stcli

    # Sisa kode tetap sama
    sys.argv = ["streamlit", "run", sys.argv[0]]
import tempfile

import streamlit as st
import sys
import os



# GANTI FUNGSI PDF LAMA DENGAN VERSI LENGKAP INI


# TAMBAHKAN DUA BARIS INI DI ATAS UNTUK MEMBERSIHKAN CACHE
st.cache_data.clear()
st.cache_resource.clear()

def resource_path(relative_path):
    """ Mendapatkan path absolut ke resource, berfungsi untuk dev dan PyInstaller """
    try:
        # PyInstaller membuat folder temp _MEIPASS saat .exe berjalan
        base_path = sys._MEIPASS
    except Exception:
        # Saat berjalan sebagai script .py biasa, gunakan path normal
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


import pandas as pd
import numpy as np





# ----- FUNGSI HELPER -----

# Letakkan fungsi ini di bagian FUNGSI HELPER
def visualize_attention_as_text(tokens, scores):
    """Menghasilkan HTML untuk menyorot teks berdasarkan skor atensi."""
    # Normalisasi skor agar nilainya antara 0 dan 1 untuk intensitas warna
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    html_parts = []
    for token, score in zip(tokens, normalized_scores):
        # HSL: Hue(0=red), Saturation(100%), Lightness(100% - score*50%)
        # Semakin tinggi skor, semakin gelap warnanya (lightness menurun)
        color = f"hsl(0, 100%, {100 - score * 50}%)"
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 3px;">{token.replace("##", "")}</span>')

    return " ".join(html_parts)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Tambahkan fungsi GELU ini di bagian FUNGSI HELPER
def gelu(x):
    """Implementasi numerik dari fungsi aktivasi GELU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, epsilon=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + epsilon) + beta


def numpy_to_latex(arr, precision=2):
    if len(arr.shape) == 1: arr = arr.reshape(1, -1)
    rows = [" & ".join(map(lambda x: f"{x:.{precision}f}", row)) for row in arr]
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def tokens_to_latex_labels(tokens):
    cleaned_tokens = [token.replace('_', r'\_') for token in tokens]
    formatted_tokens = [f"\\text{{{token}}}" for token in cleaned_tokens]
    return r"\begin{matrix}" + r" \\ ".join(formatted_tokens) + r"\end{matrix}"


# ----- FUNGSI INTI MODEL -----

@st.cache_resource
def load_model_and_tokenizer(path=resource_path("finetuned_indobert_negation_combine")):
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    try:
        if not os.path.isdir(path):
            return None, None, f"Direktori tidak ditemukan di path: '{path}'. Pastikan path sudah benar."
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        # model.half()  # Menggunakan presisi float16 untuk hemat memori
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


# GANTI FUNGSI LAMA ANDA DENGAN VERSI LENGKAP INI
@st.cache_data
def load_negation_tokens(filepath=resource_path("token_negasi.txt")):
    """Membaca file teks dan memuat token negasi ke dalam sebuah set."""
    try:
        with open(filepath, "r") as f:
            # Menggunakan set untuk pencarian yang efisien (O(1) average time complexity)
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        st.error(f"File token negasi '{filepath}' tidak ditemukan. Pastikan file berada di folder yang sama.")
        return set()

def run_full_forward_pass(_model_path, token_ids):
    import torch
    # Muat model untuk mendapatkan konfigurasi dan bobot
    _, model, _ = load_model_and_tokenizer(_model_path)
    all_weights = get_model_weights(model)

    # Dapatkan konfigurasi penting dari model
    config = model.config
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads

    # --- PERUBAHAN 1: Ambil bobot dari layer embedding ---
    embedding_weights = {
        "word_embeddings": model.bert.embeddings.word_embeddings.weight.data.numpy(),
        "position_embeddings": model.bert.embeddings.position_embeddings.weight.data.numpy(),
    "token_type_embeddings": model.bert.embeddings.token_type_embeddings.weight.data.numpy(),
        "ln_gamma": model.bert.embeddings.LayerNorm.weight.data.numpy(),
        "ln_beta": model.bert.embeddings.LayerNorm.bias.data.numpy()
    }

    with torch.no_grad():
        initial_embeddings = model.bert.embeddings(torch.tensor(token_ids).unsqueeze(0)).numpy()[0]

    hidden_states = initial_embeddings
    all_hidden_states = [initial_embeddings]
    all_attention_weights = []
    all_inspection_data = []

    for i, layer_weights in enumerate(all_weights):
        current_input = hidden_states
        seq_len = current_input.shape[0]

        # --- Logika Multi-Head Attention ---

        # 1. Proyeksi Linear Awal
        Q_proj = current_input @ layer_weights['q_w']
        K_proj = current_input @ layer_weights['k_w']
        V_proj = current_input @ layer_weights['v_w']

        # 2. Reshape untuk Multi-Head
        Q_multi_head = Q_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)
        K_multi_head = K_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)
        V_multi_head = V_proj.reshape(seq_len, num_heads, head_size).transpose(1, 0, 2)

        # 3. Perhitungan Skor Atensi Paralel
        attention_scores = Q_multi_head @ K_multi_head.transpose(0, 2, 1)
        scaled_attention_scores_multi_head = attention_scores / np.sqrt(head_size)
        attention_weights_multi_head = softmax(scaled_attention_scores_multi_head)

        # Simpan bobot atensi rata-rata dari semua kepala untuk visualisasi
        all_attention_weights.append(attention_weights_multi_head.mean(axis=0))

        # 4. Hitung Output Atensi Paralel
        attention_output_multi_head = attention_weights_multi_head @ V_multi_head

        # 5. Gabungkan (Concatenate) dan Reshape
        attention_output_concat = attention_output_multi_head.transpose(1, 0, 2).reshape(seq_len, config.hidden_size)

        # 6. Proyeksi Linear Akhir (W_o)
        attention_output_final = attention_output_concat @ layer_weights['attention_output_w'] + layer_weights[
            'attention_output_b']

        # --- Add & Norm, FFN ---

        sum_1 = current_input + attention_output_final
        add_norm_1_output = layer_norm(sum_1, layer_weights['ln1_gamma'], layer_weights['ln1_beta'])

        ffn_intermediate = add_norm_1_output @ layer_weights['ffn_w1'] + layer_weights['ffn_b1']
        ffn_activated = gelu(ffn_intermediate)
        ffn_output_raw = ffn_activated @ layer_weights['ffn_w2'] + layer_weights['ffn_b2']

        sum_2 = add_norm_1_output + ffn_output_raw
        hidden_states = layer_norm(sum_2, layer_weights['ln2_gamma'], layer_weights['ln2_beta'])
        all_hidden_states.append(hidden_states)

        # --- PENGISIAN ULANG INSPECTION_DATA SECARA LENGKAP ---
        current_layer_inspection_data = {
            "input": current_input,
            "Q": Q_proj,  # Menggunakan matriks sebelum di-reshape
            "K": K_proj,  # Menggunakan matriks sebelum di-reshape
            "V": V_proj,  # Menggunakan matriks sebelum di-reshape
            # Untuk UI, kita tampilkan rata-rata dari semua kepala
            "scaled_attention_scores": scaled_attention_scores_multi_head,
            "attention_weights": attention_weights_multi_head,
            # Variabel ini adalah hasil gabungan sebelum proyeksi linear akhir
            "attention_output": attention_output_concat,
            "attention_output_final": attention_output_final,
            "sum_1": sum_1,
            "add_norm_1_output": add_norm_1_output,
            "ffn_intermediate": ffn_intermediate,
            "ffn_activated": ffn_activated,  # Menggantikan 'ffn_relu'
            "ffn_output": ffn_output_raw,
            "sum_2": sum_2,
            "final_output": hidden_states,
            "head_size": head_size
        }
        all_inspection_data.append(current_layer_inspection_data)

    final_cls_output = hidden_states[0]
    return final_cls_output, all_inspection_data, all_weights, all_hidden_states, all_attention_weights, embedding_weights


def get_model_weights(_model):
    weights = []
    for layer in _model.bert.encoder.layer:
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
        weights.append(layer_weights)
    return weights


def get_classifier_weights(_model):
    # Mengambil bobot dari layer classifier
    return {
        "W": _model.classifier.weight.data.numpy().T,
        "b": _model.classifier.bias.data.numpy()
    }


# ----- UI APLIKASI STREAMLIT -----

st.set_page_config(layout="wide")
st.title("🔬 Analisis Atensi & Perhitungan Detail Transformer")
st.header("1. Pengaturan Model dan Input")
model_path = st.text_input("Masukkan Path Direktori Model Anda:", resource_path("finetuned_indobert_negation_combine"))
kalimat = st.text_input("Masukkan Kalimat Studi Kasus:",
                        "aku lelah merasa tidak_berharga")
# Membuat daftar pilihan untuk dropdown
layer_options = list(range(1, 13))

# Menggunakan st.selectbox sebagai pengganti st.slider
detail_layer = st.selectbox(
    "Pilih Layer untuk Analisis Detail:",
    options=layer_options,
    index=11  # Atur default ke Layer 12 (karena 12 ada di indeks ke-11)
)
detail_layer_idx = detail_layer - 1  # Baris ini tetap sama dan masih diperlukan
show_details = st.checkbox("Tampilkan Contoh Perhitungan Spesifik (memperlambat aplikasi)")

if st.button("🚀 Proses dan Analisis Kalimat"):
    from fpdf import FPDF
    import matplotlib.pyplot as plt


    class PDF(FPDF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_counter = 0
            # Mendaftarkan font proporsional (untuk teks biasa)
            self.add_font("DejaVu", "", resource_path("DejaVuSans.ttf"))
            self.add_font("DejaVu", "B", resource_path("DejaVuSans-Bold.ttf"))
            self.add_font("DejaVu", "I", resource_path("DejaVuSans-Oblique.ttf"))
            self.font_family_regular = "DejaVu"

            # Mendaftarkan font monospaced (untuk tabel/data)
            self.add_font("DejaVuMono", "", resource_path("DejaVuSansMono.ttf"))
            self.add_font("DejaVuMono", "B", resource_path("DejaVuSansMono-Bold.ttf"))
            self.font_family_mono = "DejaVuMono"

        def header(self):
            self.set_font(self.font_family_regular, 'B', 12)
            self.cell(0, 10, 'Rincian Lengkap Perhitungan Matematis', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_font(self.font_family_regular, 'I', 8)
            self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

        def section_title(self, title):
            self.set_font(self.font_family_regular, 'B', 12)
            self.cell(0, 6, title, 0, 1, 'L')
            self.ln(2)

        def subsection_title(self, title):
            self.set_font(self.font_family_regular, 'B', 10)
            self.cell(0, 5, title, 0, 1, 'L')
            self.ln(1)

        def body_text(self, text):
            self.set_font(self.font_family_regular, '', 10)
            self.multi_cell(0, 5, text)
            self.ln(2)

        def write_latex(self, formula, font_size=12):
            # ... (fungsi ini sudah benar dan tidak perlu diubah) ...
            fig, ax = plt.subplots(figsize=(6, 1), dpi=200)
            ax.text(0.5, 0.5, f'${formula}$', size=font_size, ha='center', va='center')
            ax.axis('off')

            temp_filename = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    plt.savefig(tmpfile, format='png', bbox_inches='tight', pad_inches=0.1)
                    temp_filename = tmpfile.name
                plt.close(fig)
                self.image(temp_filename, x=self.get_x() + 10, w=self.w - 40)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            self.ln(2)

        def write_dataframe(self, df, tokens=None, limit=4):
            """Menulis DataFrame, memotong kolom jika terlalu lebar."""
            self.set_font(self.font_family_mono, '', 8)

            matrix = df.to_numpy()
            num_rows, num_cols = matrix.shape

            # Jika matriks tidak terlalu lebar, tampilkan seperti biasa
            if num_cols <= (2 * limit) + 1:
                df_text = df.to_string()
                self.multi_cell(0, 4, df_text)
            # Jika terlalu lebar, potong dan tampilkan ringkasannya
            else:
                lines = []
                for i in range(num_rows):
                    token_label = tokens[i] if tokens and i < len(tokens) else str(df.index[i])
                    token_prefix = f"{token_label:<15}"

                    start_slice = matrix[i, :limit]
                    end_slice = matrix[i, -limit:]

                    start_str = " ".join([f"{val:6.2f}" for val in start_slice])
                    end_str = " ".join([f"{val:6.2f}" for val in end_slice])

                    full_row_str = f"{start_str}  ...  {end_str}"
                    lines.append(token_prefix + full_row_str)

                final_text = "\n".join(lines)
                self.multi_cell(0, 4, final_text)

            self.ln(2)


    def generate_full_math_pdf(
            kalimat, tokens, token_ids, inputs, embedding_weights, all_weights,
            all_inspection_data, detail_layer, detail_layer_idx, show_details,
            model_config, classifier_weights, logits, probabilities, labels
    ):
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        selected_layer_data = all_inspection_data[detail_layer_idx]

        # --- BAGIAN 1: EMBEDDING (Sesuai Streamlit) ---
        pdf.section_title("Langkah 1: Teks Mentah ke Final Embedding (Input H⁰)")
        pdf.body_text("Proses 'penerjemahan' dari teks menjadi matriks angka sebelum diproses oleh Layer 1.")

        pdf.subsection_title("1a. Tokenisasi & Konversi ke ID")
        token_df = pd.DataFrame(
            {'Token': tokens, 'Token ID': token_ids, 'Segment ID': inputs['token_type_ids'][0].tolist()})
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4, token_df.to_string())
        pdf.ln(2)

        word_embedding_vectors = embedding_weights["word_embeddings"][token_ids]
        position_ids = np.arange(len(token_ids))
        position_embedding_vectors = embedding_weights["position_embeddings"][position_ids]
        token_type_ids = inputs['token_type_ids'][0].numpy()
        segment_embedding_vectors = embedding_weights["token_type_embeddings"][token_type_ids]
        summed_embeddings = word_embedding_vectors + position_embedding_vectors + segment_embedding_vectors
        final_embeddings_h0 = layer_norm(summed_embeddings, embedding_weights['ln_gamma'], embedding_weights['ln_beta'])

        pdf.subsection_title("1b. Word Embedding (E_word)")
        pdf.write_dataframe(pd.DataFrame(word_embedding_vectors), tokens=tokens)

        pdf.subsection_title("1c. Position Embedding (E_pos)")
        pdf.write_dataframe(pd.DataFrame(position_embedding_vectors), tokens=tokens)

        pdf.subsection_title("1d. Segment Embedding (E_seg)")
        pdf.write_dataframe(pd.DataFrame(segment_embedding_vectors), tokens=tokens)

        pdf.subsection_title("1e. Penjumlahan, Normalisasi, dan Final Embedding (H⁰)")
        pdf.write_latex(r"H^{(0)} = \text{LayerNorm}(E_{word} + E_{pos} + E_{seg})")
        pdf.body_text("Hasil Matriks Embedding Final H^(0):")
        pdf.write_dataframe(pd.DataFrame(final_embeddings_h0), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk H^(0)[0,0]")
            e_word_00 = word_embedding_vectors[0, 0]
            e_pos_00 = position_embedding_vectors[0, 0]
            e_seg_00 = segment_embedding_vectors[0, 0]
            sum_00 = summed_embeddings[0, 0]
            pdf.write_latex(
                rf"\text{{1. Penjumlahan: }} {e_word_00:.2f} + {e_pos_00:.2f} + {e_seg_00:.2f} = {sum_00:.2f}")

            sum_row_0 = summed_embeddings[0]
            gamma_vector = embedding_weights['ln_gamma']
            beta_vector = embedding_weights['ln_beta']
            h0_00 = final_embeddings_h0[0, 0]
            epsilon = 1e-6
            mean_val = np.mean(sum_row_0)
            std_val = np.std(sum_row_0)
            pdf.body_text(f"2. Normalisasi (Mean μ={mean_val:.4f}, Std Dev σ={std_val:.4f}):")
            formula = rf"H^{{(0)}}[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - ({mean_val:.4f})}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + ({beta_vector[0]:.2f}) = {h0_00:.2f}"
            pdf.write_latex(formula)

        # --- BAGIAN 2: TRANSFORMER LAYER (Sesuai Streamlit) ---
        pdf.add_page()
        pdf.section_title(f"Perhitungan Detail untuk Layer {detail_layer}")

        pdf.subsection_title("Langkah 2: Input H dan Proyeksi Q, K, V")
        pdf.write_latex(r"Q = H \cdot W_Q \quad | \quad K = H \cdot W_K \quad | \quad V = H \cdot W_V")
        pdf.body_text(
            "Matriks input dari layer sebelumnya (H) diproyeksikan menjadi tiga matriks: Query (Q), Key (K), dan Value (V).")

        pdf.body_text(f"Matriks Input H^({detail_layer - 1}):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['input']), tokens=tokens)

        # --- [KODE BARU DITAMBAHKAN DI SINI] ---
        pdf.body_text("Hasil Matriks Q (Query):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['Q']), tokens=tokens)

        pdf.body_text("Hasil Matriks K (Key):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['K']), tokens=tokens)

        pdf.body_text("Hasil Matriks V (Value):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['V']), tokens=tokens)
        # --- [AKHIR DARI KODE BARU] ---

        pdf.subsection_title("Matriks Bobot (Weights) untuk Q, K, V (slice 5x5)")
        wq_slice = pd.DataFrame(all_weights[detail_layer_idx]['q_w'][:5, :5])
        wk_slice = pd.DataFrame(all_weights[detail_layer_idx]['k_w'][:5, :5])
        wv_slice = pd.DataFrame(all_weights[detail_layer_idx]['v_w'][:5, :5])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W_q (slice):\n{wq_slice.to_string(float_format='%.2f')}\n\nW_k (slice):\n{wk_slice.to_string(float_format='%.2f')}\n\nW_v (slice):\n{wv_slice.to_string(float_format='%.2f')}")
        pdf.ln(2)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk Q, K, V [0,0]")
            h_row0 = selected_layer_data['input'][0]
            wq_col0 = all_weights[detail_layer_idx]['q_w'][:, 0]
            q_val00 = selected_layer_data['Q'][0, 0]
            q_terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_row0, wq_col0)]
            pdf.write_latex(rf"Q[0,0] = {q_terms[0]} + \dots = {q_val00:.2f}")

            wk_col0 = all_weights[detail_layer_idx]['k_w'][:, 0]
            k_val00 = selected_layer_data['K'][0, 0]
            k_terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_row0, wk_col0)]
            pdf.write_latex(rf"K[0,0] = {k_terms[0]} + \dots = {k_val00:.2f}")

            wv_col0 = all_weights[detail_layer_idx]['v_w'][:, 0]
            v_val00 = selected_layer_data['V'][0, 0]
            v_terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_row0, wv_col0)]
            pdf.write_latex(rf"V[0,0] = {v_terms[0]} + \dots = {v_val00:.2f}")

        pdf.subsection_title("Langkah 3: Perhitungan Skor Atensi")
        pdf.write_latex(r"\text{Scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}")

        # Tentukan kepala atensi yang akan ditampilkan (konsisten dengan contoh)
        head_idx_to_show = 0

        pdf.body_text(f"Hasil Matriks Skor Atensi (dari Kepala ke-{head_idx_to_show + 1}, sebelum Softmax):")
        scores_one_head = selected_layer_data['scaled_attention_scores'][head_idx_to_show]
        pdf.write_dataframe(pd.DataFrame(scores_one_head), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk Scores[0,0]")
            head_idx_to_show = 0
            q_vector_0 = selected_layer_data['Q'][0]
            k_vector_0 = selected_layer_data['K'][0]
            raw_score_00 = np.dot(q_vector_0, k_vector_0)
            head_size = model_config.hidden_size // model_config.num_attention_heads
            scaled_score_00 = selected_layer_data['scaled_attention_scores'][head_idx_to_show, 0, 0]

            # --- [PERUBAHAN UTAMA] ---
            # Buat beberapa suku pertama dari dot product untuk ditampilkan
            dot_product_terms = [f"({q:.2f} \cdot {k:.2f})" for q, k in zip(q_vector_0[:3], k_vector_0[:3])]

            pdf.body_text("1. Dot Product (perkalian antara baris Q[0] dan K[0]):")
            pdf.write_latex(
                rf"\text{{Skor Mentah}} = {dot_product_terms[0]} + {dot_product_terms[1]} + {dot_product_terms[2]} + \dots = {raw_score_00:.2f}")

            pdf.body_text("2. Scaling (hasil dot product dibagi dengan akar dari d_k):")
            pdf.write_latex(
                rf"\text{{Skor Akhir}} = \frac{{{raw_score_00:.2f}}}{{\sqrt{{{head_size}}}}} = {scaled_score_00:.2f}")

        pdf.subsection_title("Langkah 4: Normalisasi Skor dengan Softmax")
        pdf.write_latex(r"W_{att} = \text{softmax}(\text{Scores})")

        # Gunakan indeks kepala atensi yang sama seperti di Langkah 3
        head_idx_to_show = 0

        pdf.body_text(f"Hasil Matriks Bobot Atensi (W_att) (dari Kepala ke-{head_idx_to_show + 1}):")
        weights_one_head = selected_layer_data['attention_weights'][head_idx_to_show]
        pdf.write_dataframe(pd.DataFrame(weights_one_head), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk W_att[0,0] (dari Kepala ke-1)")
            head_idx_to_show = 0
            scores_row0 = selected_layer_data['scaled_attention_scores'][head_idx_to_show, 0]
            stable_scores = scores_row0 - np.max(scores_row0)
            exp_s0 = np.exp(stable_scores[0])
            sum_exp = np.sum(np.exp(stable_scores))
            w_att00 = selected_layer_data['attention_weights'][head_idx_to_show, 0, 0]

            # --- [PERUBAHAN UTAMA] ---
            # Buat beberapa suku pertama dari penyebut (denominator) softmax
            denominator_terms = [f"\exp({s:.2f})" for s in scores_row0[:3]]

            formula = rf"W_{{att}}[0,0] = \frac{{\exp({scores_row0[0]:.2f})}}{{{denominator_terms[0]} + {denominator_terms[1]} + {denominator_terms[2]} + \dots}} = \frac{{{exp_s0:.2f}}}{{{sum_exp:.2f}}} = {w_att00:.4f}"
            pdf.write_latex(formula)

        pdf.subsection_title("Langkah 5a: Penggabungan Kontekstual")
        pdf.write_latex(r"\text{AttentionOutput} = W_{att} \cdot V")
        pdf.body_text("Hasil Matriks AttentionOutput (sebelum proyeksi):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['attention_output']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk AttentionOutput[0,0] (dari Kepala ke-1)")
            head_idx_to_show = 0
            watt_row0 = selected_layer_data['attention_weights'][head_idx_to_show, 0]
            v_col0 = selected_layer_data['V'][:, 0]
            # Note: att_out_val00 is from the concatenated output, so this is an illustrative example of one head's contribution.
            att_out_val00 = selected_layer_data['attention_output'][0, 0]
            terms = [f"({w:.2f} \cdot {v:.2f})" for w, v in zip(watt_row0, v_col0)]
            pdf.write_latex(rf"\text{{AttOut}}[0,0] = {terms[0]} + \dots = {att_out_val00:.2f} \text{{ (ilustrasi)}}")

        pdf.subsection_title("Langkah 5b: Proyeksi Linear Akhir Atensi")
        pdf.write_latex(r"\text{AttentionOutputFinal} = \text{AttentionOutput} \cdot W_o + b_o")
        pdf.subsection_title("Matriks Bobot (W_o & b_o) (slice)")
        wo_matrix = all_weights[detail_layer_idx]['attention_output_w']
        bo_vector = all_weights[detail_layer_idx]['attention_output_b']
        wo_slice = pd.DataFrame(wo_matrix[:5, :5])
        bo_slice = pd.DataFrame(bo_vector[:5], columns=['bias'])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W_o (slice 5x5):\n{wo_slice.to_string(float_format='%.2f')}\n\nb_o (slice awal):\n{bo_slice.to_string(float_format='%.2f')}")
        pdf.ln(2)
        pdf.body_text("Hasil Matriks AttentionOutputFinal:")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['attention_output_final']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk AttOutFinal[0,0]")
            att_out_row0 = selected_layer_data['attention_output'][0]
            wo_col0 = all_weights[detail_layer_idx]['attention_output_w'][:, 0]
            bo_0 = all_weights[detail_layer_idx]['attention_output_b'][0]
            att_out_final_val00 = selected_layer_data['attention_output_final'][0, 0]
            terms = [f"({att:.2f} \cdot {w:.2f})" for att, w in zip(att_out_row0, wo_col0)]
            pdf.write_latex(
                rf"\text{{AttOutFinal}}[0,0] = ({terms[0]} + \dots) + {bo_0:.2f} = {att_out_final_val00:.2f}")

        pdf.subsection_title("Langkah 6: Add & Norm (Setelah Atensi)")
        pdf.write_latex(r"H' = \text{LayerNorm}(H_{in} + \text{AttentionOutputFinal})")
        pdf.body_text("Hasil Matriks H' (Input untuk FFN):")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['add_norm_1_output']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk H'[0,0]")
            h_in_00 = selected_layer_data['input'][0, 0]
            att_out_00 = selected_layer_data['attention_output_final'][0, 0]
            sum_00 = selected_layer_data['sum_1'][0, 0]
            pdf.write_latex(rf"\text{{1. Penjumlahan: }} {h_in_00:.2f} + {att_out_00:.2f} = {sum_00:.2f}")
            sum_row_0 = selected_layer_data['sum_1'][0]
            gamma_vector = all_weights[detail_layer_idx]['ln1_gamma']
            beta_vector = all_weights[detail_layer_idx]['ln1_beta']
            h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
            epsilon = 1e-6
            mean_val = np.mean(sum_row_0)
            std_val = np.std(sum_row_0)
            pdf.body_text(f"2. Normalisasi (Mean μ={mean_val:.4f}, Std Dev σ={std_val:.4f}):")
            formula = rf"H'[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - ({mean_val:.4f})}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + ({beta_vector[0]:.2f}) = {h_prime_00:.2f}"
            pdf.write_latex(formula)

        pdf.section_title("Langkah 7: Feed-Forward Network (FFN)")
        pdf.write_latex(r"\text{FFN}(H') = \text{GELU}(H'W_1 + b_1)W_2 + b_2")
        pdf.subsection_title("Matriks Bobot untuk FFN (slice)")
        w1_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_w1'][:5, :5])
        b1_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_b1'][:5], columns=['bias'])
        w2_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_w2'][:5, :5])
        b2_slice = pd.DataFrame(all_weights[detail_layer_idx]['ffn_b2'][:5], columns=['bias'])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W1 (slice 5x5):\n{w1_slice.to_string(float_format='%.2f')}\n\n"
                       f"b1 (slice awal):\n{b1_slice.to_string(float_format='%.2f')}\n\n"
                       f"W2 (slice 5x5):\n{w2_slice.to_string(float_format='%.2f')}\n\n"
                       f"b2 (slice awal):\n{b2_slice.to_string(float_format='%.2f')}"
                       )
        pdf.ln(2)

        pdf.subsection_title("7a. Hasil setelah Linear Pertama")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['ffn_intermediate']), tokens=tokens)
        pdf.subsection_title("7b. Hasil setelah Aktivasi GELU")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['ffn_activated']), tokens=tokens)
        pdf.subsection_title("7c. Hasil setelah Linear Kedua")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['ffn_output']), tokens=tokens)

        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail FFN")
            h_prime_row0 = selected_layer_data['add_norm_1_output'][0]
            w1_col0 = all_weights[detail_layer_idx]['ffn_w1'][:, 0]
            b1_0 = all_weights[detail_layer_idx]['ffn_b1'][0]
            ffn_inter_00 = selected_layer_data['ffn_intermediate'][0, 0]
            terms1 = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(h_prime_row0, w1_col0)]
            pdf.write_latex(rf"\text{{1. Linear 1: }} {terms1[0]} + \dots + {b1_0:.2f} = {ffn_inter_00:.2f}")

            pdf.write_latex(
                rf"\text{{2. Aktivasi GELU: GELU}}({ffn_inter_00:.2f}) \to {selected_layer_data['ffn_activated'][0, 0]:.2f}")

            ffn_activated_row0 = selected_layer_data['ffn_activated'][0]
            w2_col0 = all_weights[detail_layer_idx]['ffn_w2'][:, 0]
            b2_0 = all_weights[detail_layer_idx]['ffn_b2'][0]
            ffn_output_00 = selected_layer_data['ffn_output'][0, 0]
            terms2 = [f"({r:.2f} \cdot {w:.2f})" for r, w in zip(ffn_activated_row0, w2_col0)]
            pdf.write_latex(rf"\text{{3. Linear 2: }} {terms2[0]} + \dots + {b2_0:.2f} = {ffn_output_00:.2f}")

        pdf.subsection_title("Langkah 8: Add & Norm Final")
        pdf.write_latex(r"H_{out} = \text{LayerNorm}(H' + \text{FFN}(H'))")
        pdf.body_text("Hasil Matriks Final H_out:")
        pdf.write_dataframe(pd.DataFrame(selected_layer_data['final_output']), tokens=tokens)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan Detail untuk H_out[0,0]")
            h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
            ffn_out_00 = selected_layer_data['ffn_output'][0, 0]
            sum_2_00 = selected_layer_data['sum_2'][0, 0]
            pdf.write_latex(rf"\text{{1. Penjumlahan: }} {h_prime_00:.2f} + {ffn_out_00:.2f} = {sum_2_00:.2f}")
            sum_row_2 = selected_layer_data['sum_2'][0]
            gamma_vector_2 = all_weights[detail_layer_idx]['ln2_gamma']
            beta_vector_2 = all_weights[detail_layer_idx]['ln2_beta']
            h_out_00 = selected_layer_data['final_output'][0, 0]
            mean_val_2 = np.mean(sum_row_2)
            std_val_2 = np.std(sum_row_2)
            pdf.body_text(f"2. Normalisasi (Mean μ={mean_val_2:.4f}, Std Dev σ={std_val_2:.4f}):")
            formula_2 = rf"H_{{out}}[0,0] = {gamma_vector_2[0]:.2f} \times \frac{{{sum_2_00:.2f} - ({mean_val_2:.4f})}}{{\sqrt{{{std_val_2 ** 2:.4f} + \epsilon}}}} + ({beta_vector_2[0]:.2f}) = {h_out_00:.2f}"
            pdf.write_latex(formula_2)

        # --- BAGIAN 3: KLASIFIKASI FINAL ---
        pdf.add_page()
        pdf.section_title("Tahap Klasifikasi Final")
        prediction = labels[np.argmax(probabilities)]

        pdf.subsection_title("Langkah 9: Perhitungan Logits")
        pdf.write_latex(r"Z = h_{CLS, final} \cdot W_{classifier} + b_{classifier}")
        pdf.body_text("Bobot Klasifikasi:")
        w_cls_slice = pd.DataFrame(classifier_weights['W'][:5, :])
        b_cls = pd.DataFrame(pd.Series(classifier_weights['b']), columns=['bias'])
        pdf.set_font(pdf.font_family_mono, '', 8)
        pdf.multi_cell(0, 4,
                       f"W_classifier (slice 5x{b_cls.shape[0]}):\n{w_cls_slice.to_string(float_format='%.2f')}\n\nb_classifier:\n{b_cls.to_string(float_format='%.2f')}")
        pdf.ln(2)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk Logit Z[0]")
            final_cls_output = selected_layer_data['final_output'][0]
            terms = [f"({h:.2f} \cdot {w:.2f})" for h, w in zip(final_cls_output, classifier_weights['W'][:, 0])]
            pdf.write_latex(rf"Z[0] = ({terms[0]} + \dots) + {classifier_weights['b'][0]:.2f} = {logits[0]:.2f}")

        pdf.subsection_title("Langkah 10: Probabilitas Akhir")
        pdf.write_latex(r"P_k = \frac{e^{Z_k}}{\sum_{j} e^{Z_j}}")
        pdf.body_text("Hasil Probabilitas:")
        prob_str = ""
        for label, prob in zip(labels, probabilities):
            prob_str += f"- {label}: {prob:.2%}\n"
        pdf.body_text(prob_str)
        if show_details:
            pdf.subsection_title("✍️ Contoh Perhitungan untuk Probabilitas")
            exp_z0 = np.exp(logits[0])
            sum_exp_final = np.sum(np.exp(logits))
            prob0 = probabilities[0]
            formula = rf"P(\text{{{labels[0]}}}) = \frac{{\exp({logits[0]:.2f})}}{{\dots}} = \frac{{{exp_z0:.2f}}}{{{sum_exp_final:.2f}}} = {prob0:.4f}"
            pdf.write_latex(formula)

        pdf.subsection_title("Keputusan Akhir")
        pdf.set_font(pdf.font_family_regular, 'B', 12)
        pdf.cell(0, 10, f'Prediksi Model: {prediction}', 0, 1, 'L')

        return bytes(pdf.output(dest='S'))

    tokenizer, model, error = load_model_and_tokenizer(model_path)
    if error:
        st.error(f"Gagal memuat model. **Pesan Error:** {error}")
    else:
        st.success(f"Model dari direktori '{model_path}' berhasil dimuat.")
        inputs = tokenizer(kalimat, return_tensors="pt")
        token_ids = inputs['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        st.markdown("**Hasil Tokenisasi:**");
        st.info(f"`{tokens}`")

        # --- [BLOK BARU] PENGENALAN POLA NEGASI ---
        st.markdown("---")
        st.subheader("✔️ Pengenalan Pola Negasi Khusus")

        # Muat daftar token negasi dari file Anda
        negation_tokens_set = load_negation_tokens()

        # Cari token dari kalimat input yang ada di dalam daftar Anda
        found_patterns = [token for token in tokens if token in negation_tokens_set]

        if found_patterns:
            # Jika ada token yang ditemukan, tampilkan
            display_text = ", ".join(f"`{pattern}`" for pattern in found_patterns)
            st.success(f"Pola negasi khusus terdeteksi di dalam kalimat: {display_text}")
        else:
            # Jika tidak ada, beri tahu pengguna
            st.info("Tidak ada pola negasi khusus dari daftar Anda yang terdeteksi di dalam kalimat ini.")
        # --- [AKHIR BLOK BARU] ---

        with st.spinner("Mensimulasikan forward pass dan mengumpulkan data dari semua layer..."):
            final_cls_output, all_inspection_data, all_weights, all_hidden_states, all_attention_weights, embedding_weights = run_full_forward_pass(
                model_path, tuple(token_ids))
            classifier_weights = get_classifier_weights(model)
            logits = final_cls_output @ classifier_weights['W'] + classifier_weights['b']
            probabilities = softmax(logits)
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            prediction = labels[np.argmax(probabilities)]

            # --- DEFINISIKAN FILTER DI SINI (SEBELUM HEADER 2) ---
            tokens_to_ignore = {'[cls]', '[sep]', '[pad]', '.', ',', '?', '!', ':', ';'}
            # --- BLOK KODE YANG DIPERBARUI UNTUK HEADER 2 ---
            st.header("2. Rangkuman Fokus Atensi [CLS] di Setiap Layer")
            st.markdown(
                "Bagian ini menunjukkan token **kata** mana (selain `[CLS]` dan tanda baca) yang menjadi fokus utama di setiap layer.")

            # --- PENJELASAN FASE DIMASUKKAN DI SINI ---

            st.subheader("Fase 1: Analisis Sintaksis & Konteks Dasar (Layer 1-4)")
            st.markdown(
                "Model fokus pada hubungan gramatikal dan makna dasar kata berdasarkan kata di sekitarnya.")

            st.subheader("Fase 2: Pembangunan Makna Semantik (Layer 5-8)")
            st.markdown(
                "Model mulai merangkai makna yang lebih dalam dan mengidentifikasi konsep emosional atau niat dalam kalimat.")
            st.subheader("Fase 3: Kontekstualisasi & Kesimpulan (Layer 9-12)")
            st.markdown(
                "Model mengunci sinyal paling penentu dan mengumpulkan informasi")

            for i, attention_matrix in enumerate(all_attention_weights):
                # Buat daftar token dan skor yang sudah difilter untuk layer ini
                filtered_layer_tokens = []
                filtered_layer_scores = []
                # Gunakan 'tokens' asli dan 'attention_matrix' dari layer saat ini
                for token, score in zip(tokens, attention_matrix[0]):
                    if token.lower() not in tokens_to_ignore:
                        clean_token = token.replace('##', '')
                        filtered_layer_tokens.append(clean_token)
                        filtered_layer_scores.append(score)

                # Cari token dengan skor tertinggi HANYA dari daftar yang sudah difilter
                if filtered_layer_tokens:  # Pastikan daftar tidak kosong setelah difilter
                    max_index = np.argmax(filtered_layer_scores)
                    most_attended_token = filtered_layer_tokens[max_index]
                    max_score = filtered_layer_scores[max_index]
                    st.markdown(
                        f"- **Layer {i + 1}**: Fokus utama pada token **`{most_attended_token}`** (skor: {max_score:.2f})")
            # --- AKHIR DARI BLOK YANG DIPERBARUI ---

            st.header(f"3. Analisis Atensi Detail (Layer {detail_layer})")
            cls_attention_selected_layer = all_attention_weights[detail_layer_idx][0]

            # Buat daftar baru yang sudah bersih untuk grafik (logika yang sama seperti di atas)
            filtered_tokens = []
            filtered_scores = []

            # Loop melalui token dan skor asli
            for token, score in zip(tokens, cls_attention_selected_layer):
                if token.lower() not in tokens_to_ignore:
                    clean_token = token.replace('##', '')
                    filtered_tokens.append(clean_token)
                    filtered_scores.append(score)

            # Buat grafik menggunakan data yang sudah difilter
            fig, ax = plt.subplots(figsize=(10, 4));

            ax.bar(filtered_tokens, filtered_scores, color='skyblue')

            ax.set_ylabel("Skor Atensi");
            ax.set_title(f"Bobot Atensi dari [CLS] ke Token Relevan (Layer {detail_layer})")
            plt.xticks(rotation=45, ha="right");
            plt.tight_layout()
            st.pyplot(fig)

            # Cara memanggilnya di UI Anda (di Header 3, setelah membuat grafik bar)
            st.subheader("Visualisasi Atensi dalam Kalimat")
            html_output = visualize_attention_as_text(np.array(filtered_tokens), np.array(filtered_scores))
            st.markdown(html_output, unsafe_allow_html=True)


            # GANTI SELURUH BLOK EXPANDER ANDA DENGAN INI
            st.header(f"5. Rincian Perhitungan Matematis (Layer {detail_layer})")

            # --- BLOK TOMBOL UNDUH PDF LENGKAP ---
            st.markdown("Unduh **seluruh rincian perhitungan matematis** yang ditampilkan di bawah dalam format PDF.")

            # Panggil fungsi generator PDF yang baru dan lengkap
            # Panggil fungsi generator PDF dengan argumen LENGKAP
            pdf_data = generate_full_math_pdf(
                kalimat=kalimat,
                tokens=tokens,
                token_ids=token_ids,
                inputs=inputs,
                embedding_weights=embedding_weights,
                all_weights=all_weights,
                all_inspection_data=all_inspection_data,
                detail_layer=detail_layer,
                detail_layer_idx=detail_layer_idx,
                model_config=model.config,
                show_details=show_details,
                # --- TAMBAHKAN 4 ARGUMEN YANG HILANG INI ---
                classifier_weights=classifier_weights,
                logits=logits,
                probabilities=probabilities,
                labels=labels
            )

            st.download_button(
                label="📥 Unduh Perhitungan Lengkap (PDF)",
                data=pdf_data,
                file_name=f"perhitungan_lengkap_layer_{detail_layer}.pdf",
                mime="application/pdf"
            )
            # --- AKHIR BLOK TOMBOL UNDUH ---

            with st.expander(
                    f"Klik untuk melihat semua rumus, matriks, dan contoh perhitungan untuk Layer {detail_layer}"):
                # Ambil data inspeksi untuk layer yang dipilih dari slider (sudah benar)
                selected_layer_data = all_inspection_data[detail_layer_idx]

                # GANTI BAGIAN AWAL DI DALAM EXPANDER ANDA DENGAN BLOK KODE LENGKAP INI

                # GANTI BAGIAN AWAL DI DALAM EXPANDER ANDA DENGAN BLOK KODE LENGKAP INI

                st.subheader("Langkah 1: Teks Mentah ke Final Embedding (Input H⁰)")
                st.markdown(
                    "Ini adalah proses 'penerjemahan' dari teks yang bisa dibaca manusia menjadi matriks angka yang dipahami model, sebelum diproses oleh Layer 1.")

                # --- 1a. Tokenisasi & Konversi ke ID ---
                st.markdown("---");
                st.markdown("#### 1a. Tokenisasi & Konversi ke ID")
                st.markdown("Kalimat dipecah menjadi token, dan setiap token dipetakan ke sebuah nomor ID unik.")
                token_df = pd.DataFrame(
                    {'Token': tokens, 'Token ID': token_ids, 'Segment ID': inputs['token_type_ids'][0].tolist()})
                st.dataframe(token_df)

                # --- 1b. Word Embeddings ---
                st.markdown("---");
                st.markdown("#### 1b. Pembentukan Word Embedding")
                st.markdown("Setiap Token ID digunakan untuk mengambil vektor maknanya dari matriks embedding.")
                word_embedding_vectors = embedding_weights["word_embeddings"][token_ids]
                st.write(f"**Vektor Word Embedding (E_word) untuk kalimat Anda:**")
                st.dataframe(word_embedding_vectors.round(2))

                # --- 1c. Position Embeddings ---
                st.markdown("---");
                st.markdown("#### 1c. Pembentukan Position Embedding")
                st.markdown("Vektor posisi ditambahkan untuk memberi informasi urutan kata pada model.")
                position_ids = np.arange(len(token_ids))
                position_embedding_vectors = embedding_weights["position_embeddings"][position_ids]
                st.write(f"**Vektor Position Embedding (E_pos) untuk kalimat Anda:**")
                st.dataframe(position_embedding_vectors.round(2))

                # --- 1d. Segment Embeddings ---
                st.markdown("---");
                st.markdown("#### 1d. Pembentukan Segment Embedding")
                st.markdown(
                    "Vektor segmen ditambahkan untuk membedakan antar kalimat (dalam kasus ini, semua adalah segmen 0).")
                token_type_ids = inputs['token_type_ids'][0].numpy()

                segment_embedding_vectors = embedding_weights["token_type_embeddings"][token_type_ids]
                st.write(f"**Vektor Segment Embedding (E_seg) untuk kalimat Anda:**")
                st.dataframe(segment_embedding_vectors.round(2))

                # --- 1e. Penjumlahan & Normalisasi ---
                st.markdown("---");
                st.markdown("#### 1e. Penjumlahan, Normalisasi, dan Final Embedding (H⁰)")
                st.markdown("Ketiga vektor embedding dijumlahkan, lalu distabilkan dengan Layer Normalization.")
                st.latex(r"H^{(0)} = \text{LayerNorm}(E_{word} + E_{pos} + E_{seg})")

                # Lakukan penjumlahan dan normalisasi manual
                summed_embeddings = word_embedding_vectors + position_embedding_vectors + segment_embedding_vectors
                final_embeddings_h0 = layer_norm(summed_embeddings, embedding_weights['ln_gamma'],
                                                 embedding_weights['ln_beta'])

                st.markdown("**Hasil Matriks Embedding Final `H^(0)` (Input untuk Layer 1):**")
                col_f1, col_f2 = st.columns([1, 10]);
                with col_f1:
                    st.latex(tokens_to_latex_labels(tokens))
                with col_f2:
                    st.latex(f"H^{{(0)}} = {numpy_to_latex(final_embeddings_h0, 2)}")

                # --- TAMBAHAN: Contoh Perhitungan Detail untuk Embedding ---
                if show_details:
                    st.markdown("✍️ **Contoh Perhitungan Detail untuk Elemen Pertama `H^(0)[0,0]`**:");

                    # 1. Contoh Penjumlahan
                    st.markdown("**1. Penjumlahan Embedding (element-wise):**")
                    e_word_00 = word_embedding_vectors[0, 0]
                    e_pos_00 = position_embedding_vectors[0, 0]
                    e_seg_00 = segment_embedding_vectors[0, 0]
                    sum_00 = summed_embeddings[0, 0]
                    st.latex(rf"Sum[0,0] = E_{{word}}[0,0] + E_{{pos}}[0,0] + E_{{seg}}[0,0]")
                    st.latex(rf"{sum_00:.2f} = {e_word_00:.2f} + {e_pos_00:.2f} + {e_seg_00:.2f}")

                    # 2. Contoh Layer Normalization
                    st.markdown("**2. Normalisasi (LayerNorm):**")
                    st.markdown(
                        "Fungsi `LayerNorm` diterapkan pada **seluruh baris pertama** dari matriks hasil penjumlahan di atas.")
                    sum_row_0 = summed_embeddings[0]
                    gamma_vector = embedding_weights['ln_gamma']
                    beta_vector = embedding_weights['ln_beta']
                    h0_00 = final_embeddings_h0[0, 0]
                    epsilon = 1e-6

                    mean_val = np.mean(sum_row_0)
                    std_val = np.std(sum_row_0)

                    st.markdown(f"   - **Mean (μ) dari seluruh elemen di baris pertama:** `{mean_val:.4f}`")
                    st.markdown(f"   - **Standar Deviasi (σ) dari seluruh elemen di baris ini:** `{std_val:.4f}`")

                    st.latex(r"H^{(0)}[i, j] = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j")

                    calc_str = rf"H^{{(0)}}[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - {mean_val:.4f}}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + {beta_vector[0]:.2f}"
                    calc_result = rf" = {h0_00:.2f}"
                    st.latex(calc_str + calc_result)

                # --- AKHIR DARI BLOK PENGGANTI ---

                st.subheader("Langkah 1 & 2: Input H dan Hasil Q, K, V")
                st.latex(r"Q = H \cdot W_Q \quad | \quad K = H \cdot W_K \quad | \quad V = H \cdot W_V")
                st.markdown(
                    f"**Matriks Input `H^{{({detail_layer - 1})}}` dan Hasil Matriks `Q`, `K`, `V` untuk Layer {detail_layer}:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Input H**");
                    st.dataframe(selected_layer_data['input'].round(2))
                with col2:
                    st.markdown("**Hasil Q**");
                    st.dataframe(selected_layer_data['Q'].round(2))
                with col3:
                    st.markdown("**Hasil K**");
                    st.dataframe(selected_layer_data['K'].round(2))
                with col4:
                    st.markdown("**Hasil V**");
                    st.dataframe(selected_layer_data['V'].round(2))

                st.markdown("**Matriks Bobot (Weights) dari Model (ditampilkan sebagian 5x5):**")
                # --- PERBAIKAN 1: Gunakan detail_layer_idx untuk mengambil bobot yang benar ---
                wq_matrix = all_weights[detail_layer_idx]['q_w']
                wk_matrix = all_weights[detail_layer_idx]['k_w']
                wv_matrix = all_weights[detail_layer_idx]['v_w']

                w_col1, w_col2, w_col3 = st.columns(3)
                with w_col1:
                    st.write("$W_Q$ (slice)");
                    st.dataframe(wq_matrix[:5, :5].round(2))
                with w_col2:
                    st.write("$W_K$ (slice)");
                    st.dataframe(wk_matrix[:5, :5].round(2))
                with w_col3:
                    st.write("$W_V$ (slice)");
                    st.dataframe(wv_matrix[:5, :5].round(2))

                # --- PERBAIKAN 2: Blok duplikat yang menampilkan H, Q, K, V lagi dihapus ---

                # --- PERBAIKAN 3: Gunakan selected_layer_data untuk semua perhitungan detail ---
                if show_details:
                    h_row0 = selected_layer_data['input'][0]
                    st.markdown("✍️ **Contoh Perhitungan Elemen Pertama (baris 0, kolom 0):**")

                    wq_col0 = all_weights[detail_layer_idx]['q_w'][:, 0]
                    q_val00 = selected_layer_data['Q'][0, 0]
                    terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_row0, wq_col0)]
                    st.latex(rf"Q[0,0] = {' + '.join(terms)} = {q_val00:.2f}")

                    wk_col0 = all_weights[detail_layer_idx]['k_w'][:, 0]
                    k_val00 = selected_layer_data['K'][0, 0]
                    terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_row0, wk_col0)]
                    st.latex(rf"K[0,0] = {' + '.join(terms)} = {k_val00:.2f}")

                    wv_col0 = all_weights[detail_layer_idx]['v_w'][:, 0]
                    v_val00 = selected_layer_data['V'][0, 0]
                    terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_row0, wv_col0)]
                    st.latex(rf"V[0,0] = {' + '.join(terms)} = {v_val00:.2f}")

                st.markdown("---");
                st.subheader("Langkah 4: Normalisasi Skor dengan Softmax")
                st.latex(r"W_{att} = \text{softmax}(\text{Attention Scores})")

                st.markdown("---");
                st.subheader(f"Matriks Bobot Atensi (W_att) untuk Layer {detail_layer}")
                st.markdown(
                    "Ini adalah matriks yang **benar-benar digunakan** oleh model untuk langkah selanjutnya. Model menggunakan semua matriks dari ke-12 kepala secara bersamaan.")

                # Ambil matriks atensi 3D lengkap (12, panjang_token, panjang_token)
                full_attention_matrix = selected_layer_data['attention_weights']
                num_heads = full_attention_matrix.shape[0]

                # Buat daftar nama untuk setiap tab
                tab_names = [f"Kepala {i + 1}" for i in range(num_heads)]

                # Buat container tab untuk menampilkan matriks setiap kepala
                tabs = st.tabs(tab_names)

                head_idx_for_example = 0  # Pilih Kepala 1 untuk dijadikan contoh perhitungan

                for i, tab in enumerate(tabs):
                    with tab:
                        head_matrix = full_attention_matrix[i]
                        df_head = pd.DataFrame(head_matrix, index=tokens, columns=tokens)
                        styler = df_head.style.background_gradient(cmap='viridis', axis=None).format("{:.2f}")

                        # Beri highlight jika ini adalah kepala yang digunakan untuk contoh
                        if i == head_idx_for_example:
                            st.info(
                                f"Baris pertama di-highlight karena digunakan pada contoh perhitungan detail di bawah (Kepala {i + 1}).")
                            styler = styler.set_properties(**{'background-color': '#fde047'},
                                                           subset=pd.IndexSlice[tokens[0], :])

                        st.dataframe(styler)

                # --- BAGIAN CONTOH PERHITUNGAN DETAIL (TETAP ADA JIKA DICENTANG) ---
                if show_details:
                    st.markdown("---")
                    st.subheader(f"Contoh Perhitungan Detail (Layer {detail_layer}, dari Kepala Atensi ke-1)")

                    # Tentukan kepala atensi mana yang akan digunakan sebagai contoh
                    head_idx_for_example = 0

                    # --- BAGIAN 1: PERHITUNGAN SKOR ATENSI ---
                    st.markdown("**Perhitungan Skor Atensi `Scores[0,0]`**")

                    # Ambil data yang diperlukan
                    q_vector_0 = selected_layer_data['Q'][0]
                    k_vector_0 = selected_layer_data['K'][0]
                    head_size = selected_layer_data['head_size']  # Diambil dari data inspeksi
                    scaled_score_00 = selected_layer_data['scaled_attention_scores'][head_idx_for_example, 0, 0]

                    # Hitung skor mentah
                    raw_score_00 = np.dot(q_vector_0, k_vector_0)

                    # Tampilkan perhitungan
                    st.latex(
                        rf"\text{{1. Dot Product: }} Q[0] \cdot K[0] = ({q_vector_0[0]:.2f} \cdot {k_vector_0[0]:.2f}) + \dots = {raw_score_00:.2f}")
                    st.latex(
                        rf"\text{{2. Scaling: }} \frac{{{raw_score_00:.2f}}}{{\sqrt{{{head_size}}}}} = {scaled_score_00:.2f}")

                    # --- BAGIAN 2: PERHITUNGAN BOBOT ATENSI ---
                    st.markdown("**Perhitungan Bobot Atensi `W_att[0,0]`**")

                    # Ambil baris skor dari kepala atensi yang dipilih
                    scores_row0 = selected_layer_data['scaled_attention_scores'][head_idx_for_example, 0]

                    # Terapkan "max trick" untuk stabilitas numerik
                    stable_scores = scores_row0 - np.max(scores_row0)

                    # Hitung nilai untuk contoh, menggunakan skor yang sudah stabil
                    exp_s0 = np.exp(stable_scores[0])
                    sum_exp = np.sum(np.exp(stable_scores))
                    w_att00 = selected_layer_data['attention_weights'][head_idx_for_example, 0, 0]

                    # Tampilkan formula dengan nilai yang sudah benar dan stabil
                    st.latex(
                        rf"W_{{att}}[0,0] = \frac{{\exp({scores_row0[0]:.2f})}}{{\exp({scores_row0[0]:.2f}) + \exp({scores_row0[1]:.2f}) + \dots}} = \frac{{{exp_s0:.2f}}}{{{sum_exp:.2f}}} = {w_att00:.4f}")
                st.markdown("---");
                st.subheader("Langkah 5: Menghitung Output Atensi")
                st.markdown("---");
                st.subheader("Langkah 5a: Menghitung Output Atensi Awal (AttOut)")
                st.latex(r"\text{AttentionOutput} = W_{att} \cdot V")
                st.markdown("**Hasil Matriks `AttentionOutput`:**")
                st.dataframe(selected_layer_data['attention_output'].round(2))

                if show_details:
                    if show_details:
                        st.markdown("✍️ **Contoh Perhitungan Elemen `Att_Out[0,0]` (dari Kepala Atensi ke-1)**:");

                        # 1. Pilih satu kepala atensi untuk dijadikan contoh (misal, indeks 0)
                        head_idx_to_show = 0

                        # 2. Ambil SATU BARIS dari kepala atensi yang dipilih (INI PERBAIKANNYA)
                        watt_row0 = selected_layer_data['attention_weights'][head_idx_to_show, 0]

                        # Sisa kode sudah benar
                        v_col0 = selected_layer_data['V'][:, 0]
                        att_out_val00 = selected_layer_data['attention_output'][0, 0]
                        terms = [rf"({w:.2f} \times {v:.2f})" for w, v in zip(watt_row0, v_col0)]
                        st.latex(rf"\text{{Att\_Out}}[0,0] = {' + '.join(terms)} = {att_out_val00:.2f}")

                st.markdown("---");
                st.subheader("Langkah 5b: Proyeksi Linear dari Output Atensi")
                st.markdown(
                    "Hasil atensi awal kemudian diproses oleh sebuah layer linear (Dense) untuk menghasilkan output final dari blok atensi.")
                st.latex(r"\text{AttentionOutputFinal} = \text{AttentionOutput} \cdot W_o + b_o")
                st.markdown("**Matriks Bobot untuk Proyeksi Linear (ditampilkan sebagian):**")
                wo_matrix = all_weights[detail_layer_idx]['attention_output_w']
                bo_vector = all_weights[detail_layer_idx]['attention_output_b']
                w_col1, w_col2 = st.columns(2)
                with w_col1:
                    st.write("$W_o$ (slice 5x5)")
                    st.dataframe(wo_matrix[:5, :5].round(2))
                with w_col2:
                    st.write("$b_o$ (slice awal)")
                    st.dataframe(bo_vector[:5].round(2))
                st.markdown("**Hasil Matriks `AttentionOutputFinal`:**")
                st.dataframe(selected_layer_data['attention_output_final'].round(2))

                if show_details:
                    st.markdown("✍️ **Contoh Perhitungan Elemen `AttOutFinal[0,0]`:**");
                    att_out_row0 = selected_layer_data['attention_output'][0]
                    wo_col0 = all_weights[detail_layer_idx]['attention_output_w'][:, 0]
                    bo_0 = all_weights[detail_layer_idx]['attention_output_b'][0]
                    att_out_final_val00 = selected_layer_data['attention_output_final'][0, 0]

                    terms = [rf"({att:.2f} \times {w:.2f})" for att, w in zip(att_out_row0, wo_col0)]
                    st.latex(rf"\text{{AttOutFinal}}[0,0] = ({' + '.join(terms)}) + {bo_0:.2f} = {att_out_final_val00:.2f}")

                st.markdown("---");
                # --- [BAGIAN YANG DIPERBARUI] Langkah 6: Add & Norm, lalu FFN ---
                st.markdown("---");
                st.subheader("Langkah 6: Add & Norm (Setelah Atensi)")
                st.latex(r"H' = \text{LayerNorm}(H_{in} + \text{AttentionOutput})")
                st.markdown("**Hasil Matriks `H'` (Input untuk FFN):**")
                st.dataframe(selected_layer_data['add_norm_1_output'].round(2))
                # Bagian 1: Penjumlahan (Add)
                h_in_00 = selected_layer_data['input'][0, 0]
                att_out_00 = selected_layer_data['attention_output_final'][0, 0]
                sum_00 = selected_layer_data['sum_1'][0, 0]
                st.latex(
                    rf"\text{{1. Penjumlahan: }} H_{{in}}[0,0] + \text{{AttOutFinal}}[0,0] = {h_in_00:.2f} + {att_out_00:.2f} = {sum_00:.2f}")

                # --- Bagian 2: Normalisasi (Norm) ---
                st.markdown(
                    "2. **Normalisasi**: Fungsi `LayerNorm` diterapkan pada **seluruh baris pertama** dari matriks hasil penjumlahan di atas.")

                # Ambil data yang diperlukan untuk perhitungan
                sum_row_0 = selected_layer_data['sum_1'][0]
                gamma_vector = all_weights[i]['ln1_gamma']
                beta_vector = all_weights[i]['ln1_beta']
                h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
                epsilon = 1e-6

                # Hitung mean dan std
                mean_val = np.mean(sum_row_0)
                std_val = np.std(sum_row_0)

                st.markdown(
                    f"   - **Vektor baris pertama (sebelum normalisasi) dimulai dengan:** `[{sum_row_0[0]:.2f}, {sum_row_0[1]:.2f}, {sum_row_0[2]:.2f}, ...]`")
                st.markdown(f"   - **Hitung Mean (μ) dari seluruh elemen di baris ini:** `{mean_val:.4f}`")
                st.markdown(f"   - **Hitung Standar Deviasi (σ) dari seluruh elemen di baris ini:** `{std_val:.4f}`")

                st.markdown("   - **Terapkan rumus LayerNorm untuk elemen pertama** ($H'[0,0]$):")
                st.latex(r"H'[i, j] = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j")

                calc_str = rf"H'[0,0] = {gamma_vector[0]:.2f} \times \frac{{{sum_00:.2f} - {mean_val:.4f}}}{{\sqrt{{{std_val ** 2:.4f} + \epsilon}}}} + {beta_vector[0]:.2f}"
                calc_result = rf" = {h_prime_00:.2f}"
                st.latex(calc_str + calc_result)

                st.markdown("---");
                st.markdown("---");
                st.subheader("Langkah 7: Feed-Forward Network (FFN)")
                st.latex(r"\text{FFN}(H') = \max(0, H'W_1 + b_1)W_2 + b_2")

                st.markdown("**Matriks Bobot untuk FFN (ditampilkan sebagian):**")
                w1_matrix = all_weights[i]['ffn_w1']
                b1_vector = all_weights[i]['ffn_b1']
                w2_matrix = all_weights[i]['ffn_w2']
                b2_vector = all_weights[i]['ffn_b2']
                ffn_w_col1, ffn_w_col2 = st.columns(2)
                with ffn_w_col1:
                    st.write("$W_1$ (slice 5x5)");
                    st.dataframe(w1_matrix[:5, :5].round(2))
                    st.write("$b_1$ (slice awal)");
                    st.dataframe(b1_vector[:5].round(2))
                with ffn_w_col2:
                    st.write("$W_2$ (slice 5x5)");
                    st.dataframe(w2_matrix[:5, :5].round(2))
                    st.write("$b_2$ (slice awal)");
                    st.dataframe(b2_vector[:5].round(2))

                st.markdown("**7a. Hasil setelah Linear Pertama ($H'W_1 + b_1$):**")
                st.dataframe(selected_layer_data['ffn_intermediate'].round(2))
                st.markdown("**7b. Hasil setelah Aktivasi ReLU (max(0, ...)):**")
                st.dataframe(selected_layer_data['ffn_activated'].round(2))
                st.markdown("**7c. Hasil setelah Linear Kedua (Output FFN):**")
                st.dataframe(selected_layer_data['ffn_output'].round(2))

                if show_details:
                    st.markdown("✍️ **Contoh Perhitungan Elemen Pertama di FFN (baris 0, kolom 0):**")
                    h_prime_row0 = selected_layer_data['add_norm_1_output'][0]
                    w1_col0 = all_weights[i]['ffn_w1'][:, 0]
                    b1_0 = all_weights[i]['ffn_b1'][0]
                    ffn_inter_00 = selected_layer_data['ffn_intermediate'][0, 0]
                    terms1 = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(h_prime_row0, w1_col0)]
                    st.latex(
                        rf"\text{{1. Linear Pertama: }} \text{{FFN\_inter}}[0,0] = ({' + '.join(terms1)}) + {b1_0:.2f} = {ffn_inter_00:.2f}")

                    # --- PERUBAHAN DI SINI ---
                    st.latex(
                        rf"\text{{2. Aktivasi GELU: }} \text{{GELU}}({ffn_inter_00:.2f}) \to {selected_layer_data['ffn_activated'][0, 0]:.2f}")

                    # --- DAN DI SINI ---
                    ffn_activated_row0 = selected_layer_data['ffn_activated'][0]  # Mengganti 'ffn_relu'
                    w2_col0 = all_weights[i]['ffn_w2'][:, 0]
                    b2_0 = all_weights[i]['ffn_b2'][0]
                    ffn_output_00 = selected_layer_data['ffn_output'][0, 0]

                    # --- DAN DI SINI ---
                    terms2 = [rf"({r:.2f} \times {w:.2f})" for r, w in
                              zip(ffn_activated_row0, w2_col0)]  # Menggunakan variabel baru
                    st.latex(
                        rf"\text{{3. Linear Kedua: }} \text{{FFN\_out}}[0,0] = ({' + '.join(terms2)}) + {b2_0:.2f} = {ffn_output_00:.2f}")

                    # --- [BAGIAN YANG DIPERBARUI] Langkah 8 ---
                    st.markdown("---");
                    st.subheader("Langkah 8: Add & Norm Final")
                    st.latex(r"H_{out} = \text{LayerNorm}(H' + \text{FFN}(H'))")
                    st.markdown("**Hasil Matriks Final `H_out` (Output dari Layer 1):**")
                    st.dataframe(selected_layer_data['final_output'].round(2))

                    if show_details:
                        st.markdown("✍️ **Contoh Perhitungan Elemen `H_out[0,0]`:**")

                        # Bagian 1: Penjumlahan
                        h_prime_00 = selected_layer_data['add_norm_1_output'][0, 0]
                        ffn_out_00 = selected_layer_data['ffn_output'][0, 0]
                        sum_2_00 = selected_layer_data['sum_2'][0, 0]
                        st.latex(
                            rf"\text{{1. Penjumlahan: }} H'[0,0] + \text{{FFN}}[0,0] = {h_prime_00:.2f} + {ffn_out_00:.2f} = {sum_2_00:.2f}")

                        # Bagian 2: Normalisasi (Versi Lengkap)
                        st.markdown(
                            "2. **Normalisasi**: Fungsi `LayerNorm` diterapkan pada seluruh baris pertama dari hasil penjumlahan di atas.")

                        sum_row_2 = selected_layer_data['sum_2'][0]
                        gamma_vector_2 = all_weights[i]['ln2_gamma']
                        beta_vector_2 = all_weights[i]['ln2_beta']
                        h_out_00 = selected_layer_data['final_output'][0, 0]
                        epsilon = 1e-6

                        mean_val_2 = np.mean(sum_row_2)
                        std_val_2 = np.std(sum_row_2)

                        st.markdown(f"   - **Hitung Mean (μ) dari seluruh elemen di baris ini:** `{mean_val_2:.4f}`")
                        st.markdown(
                            f"   - **Hitung Standar Deviasi (σ) dari seluruh elemen di baris ini:** `{std_val_2:.4f}`")
                        st.markdown("   - **Terapkan rumus LayerNorm untuk elemen pertama** ($H_{out}[0,0]$):")
                        st.latex(r"H_{out}[i, j] = \gamma_j \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j")

                        calc_str = rf"H_{{out}}[0,0] = {gamma_vector_2[0]:.2f} \times \frac{{{sum_2_00:.2f} - {mean_val_2:.4f}}}{{\sqrt{{{std_val_2 ** 2:.4f} + \epsilon}}}} + {beta_vector_2[0]:.2f}"
                        calc_result = rf" = {h_out_00:.2f}"
                        st.latex(calc_str + calc_result)

    # [BAGIAN YANG DIPERBARUI] Tahap Klasifikasi Final
    st.header("5. Tahap Klasifikasi Final")
    st.write("Vektor `[CLS]` dari output layer terakhir (Layer 12) digunakan untuk prediksi akhir.")
    st.latex(f"h_{{CLS, final}} = {numpy_to_latex(final_cls_output, 2)}")

    st.subheader("Langkah 9: Perhitungan Logits")
    st.latex(r"Z = h_{CLS, final} \cdot W_{classifier} + b_{classifier}")

    W_classifier = classifier_weights['W'];
    b_classifier = classifier_weights['b']
    st.markdown("**Bobot Klasifikasi (ditampilkan sebagian):**")
    col_w, col_b = st.columns(2)
    with col_w:
        st.write("$W_{classifier}$ (slice 5x2)");
        st.dataframe(W_classifier[:5, :].round(2))
    with col_b:
        st.write("$b_{classifier}$");
        st.dataframe(b_classifier.round(2))

    st.markdown("**Hasil Logits (Z):**")
    st.latex(f"Z = {numpy_to_latex(logits)}")

    if show_details:
        st.markdown("✍️ **Contoh Perhitungan Logit Pertama `Z[0]`:**")
        terms = [rf"({h:.2f} \times {w:.2f})" for h, w in zip(final_cls_output, W_classifier[:, 0])]
        st.latex(rf"Z[0] = ({' + '.join(terms)}) + {b_classifier[0]:.2f} = {logits[0]:.2f}")

    st.subheader("Langkah 10: Fungsi Softmax & Probabilitas Akhir")
    st.latex(r"P_k = \frac{e^{Z_k}}{\sum_{j} e^{Z_j}}")

    labels = ["Tidak Terindikasi", "Terindikasi Depresi"]
    prediction = labels[np.argmax(probabilities)]

    st.markdown("**Hasil Probabilitas:**")
    for i, label in enumerate(labels):
        st.markdown(f"- **{label}**: {probabilities[i]:.2%}")

    if show_details:
        st.markdown("✍️ **Contoh Perhitungan Probabilitas:**")
        exp_z0 = np.exp(logits[0]);
        exp_z1 = np.exp(logits[1])
        sum_exp = exp_z0 + exp_z1
        st.latex(
            rf"P(\text{{{labels[0]}}}) = \frac{{e^{{{logits[0]:.2f}}}}}{{e^{{{logits[0]:.2f}}} + e^{{{logits[1]:.2f}}}}} = \frac{{{exp_z0:.2f}}}{{{sum_exp:.2f}}} = {probabilities[0]:.4f}")
        st.latex(
            rf"P(\text{{{labels[1]}}}) = \frac{{e^{{{logits[1]:.2f}}}}}{{e^{{{logits[0]:.2f}}} + e^{{{logits[1]:.2f}}}}} = \frac{{{exp_z1:.2f}}}{{{sum_exp:.2f}}} = {probabilities[1]:.4f}")

    st.subheader("Keputusan Akhir")
    st.success(f"## **Prediksi Model: {prediction}**")

    # --- TAMBAHKAN BLOK RINGKASAN NARATIF DI SINI ---
    st.markdown("---")
    st.header("📜 Ringkasan Cerita Analisis")

    # Kumpulkan token-token kunci dari rangkuman Header 2
    key_tokens_per_layer = []
    for i, attention_matrix in enumerate(all_attention_weights):
        filtered_layer_tokens = []
        filtered_layer_scores = []
        for token, score in zip(tokens, attention_matrix[0]):
            if token.lower() not in tokens_to_ignore:
                clean_token = token.replace('##', '')
                filtered_layer_tokens.append(clean_token)
                filtered_layer_scores.append(score)
        if filtered_layer_tokens:
            max_index = np.argmax(filtered_layer_scores)
            key_tokens_per_layer.append(filtered_layer_tokens[max_index])
        else:
            key_tokens_per_layer.append("")  # Tambahkan string kosong jika tidak ada token

    # Buat narasi menggunakan f-string
    summary_narrative = f"""
        Secara ringkas, begitulah perjalanan "pikiran" model saat menganalisis kalimat Anda:

        1.  **Fase Awal (Layer 1-4):** Model pertama-tama membedah struktur kalimat dan dengan cepat mengidentifikasi kata kunci yang menunjukkan kondisi, seperti **`{key_tokens_per_layer[1]}`** dan **`{key_tokens_per_layer[2]}`**.

        2.  **Fase Tengah (Layer 5-8):** Selanjutnya, ia mulai membangun pemahaman yang lebih dalam. Fokusnya mulai bergeser untuk memahami konteks yang lebih luas dari kondisi tersebut, seperti yang ditunjukkan oleh perhatian pada kata **`{key_tokens_per_layer[5]}`** dan **`{key_tokens_per_layer[7]}`**.

        3.  **Fase Akhir (Layer 9-12):** Pada tahap final, model mengunci niat atau sentimen paling kuat dari kalimat. Ini terlihat dari fokusnya yang konsisten pada kata-kata penentu seperti **`{key_tokens_per_layer[8]}`** dan **`{key_tokens_per_layer[10]}`**.

        Berdasarkan akumulasi pemahaman dari awal hingga akhir inilah, model akhirnya mengambil keputusan bahwa kalimat tersebut **{prediction}** dengan probabilitas **{probabilities[np.argmax(probabilities)]:.2%}**.
        """
    st.success(summary_narrative)
    # --- AKHIR DARI BLOK RINGKASAN ---

# --- GANTI BLOK LAMA ANDA DENGAN BLOK BARU INI ---

if __name__ == '__main__':
    import multiprocessing
    import os
    import sys

    # Baris ini penting untuk mencegah error rekursi proses
    multiprocessing.freeze_support()

    # --- BAGIAN PENTING DIMULAI DI SINI ---
    # Cek apakah "flag" sudah ada. Jika ya, berarti ini adalah eksekusi kedua, jadi berhenti.
    if os.environ.get("STREAMLIT_RUNNING_MAIN"):
        sys.exit()

    # Jika ini eksekusi pertama, atur "flag" sebelum memanggil Streamlit
    os.environ["STREAMLIT_RUNNING_MAIN"] = "true"
    # --- BAGIAN PENTING BERAKHIR DI SINI ---

    import streamlit.web.cli as stcli

    # Sisa kode tetap sama
    sys.argv = ["streamlit", "run", sys.argv[0]]