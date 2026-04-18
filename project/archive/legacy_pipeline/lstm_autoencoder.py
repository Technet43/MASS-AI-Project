"""
MASS-AI: LSTM Autoencoder ile Anomali Tespiti
==============================================
Sifirdan (from scratch) NumPy ile LSTM Autoencoder implementasyonu.
Framework bagimliligi yok — saf matematik.

Yaklasim:
- Normal tuketim paternlerini ogrenir (unsupervised)
- Reconstruction error yuksek olan = anomali/kacak

Yazar: Omer Burak Kocak
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


# ============================================================
# LSTM CELL — Sifirdan Implementasyon
# ============================================================
class LSTMCell:
    """
    Tek bir LSTM hucre implementasyonu.
    
    Formüller:
    f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)     # forget gate
    i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)     # input gate
    c_hat = tanh(W_c @ [h_{t-1}, x_t] + b_c)      # candidate
    c_t = f_t * c_{t-1} + i_t * c_hat              # cell state
    o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)      # output gate
    h_t = o_t * tanh(c_t)                           # hidden state
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        concat_size = input_size + hidden_size
        
        # Forget gate
        self.W_f = np.random.randn(hidden_size, concat_size) * scale
        self.b_f = np.ones(hidden_size) * 1.0  # Bias to 1 for remembering
        
        # Input gate
        self.W_i = np.random.randn(hidden_size, concat_size) * scale
        self.b_i = np.zeros(hidden_size)
        
        # Candidate
        self.W_c = np.random.randn(hidden_size, concat_size) * scale
        self.b_c = np.zeros(hidden_size)
        
        # Output gate
        self.W_o = np.random.randn(hidden_size, concat_size) * scale
        self.b_o = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, x, h_prev, c_prev):
        """Tek adim forward pass"""
        concat = np.concatenate([h_prev, x])
        
        f = self.sigmoid(self.W_f @ concat + self.b_f)
        i = self.sigmoid(self.W_i @ concat + self.b_i)
        c_hat = np.tanh(self.W_c @ concat + self.b_c)
        c = f * c_prev + i * c_hat
        o = self.sigmoid(self.W_o @ concat + self.b_o)
        h = o * np.tanh(c)
        
        # Cache for backprop
        cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'f': f, 'i': i, 'c_hat': c_hat, 'c': c, 'o': o, 'h': h,
            'concat': concat
        }
        return h, c, cache
    
    def backward(self, dh, dc_next, cache, lr=0.001):
        """Tek adim backward pass (TBPTT - Truncated BPTT)"""
        f, i, c_hat, c, o = cache['f'], cache['i'], cache['c_hat'], cache['c'], cache['o']
        c_prev, concat = cache['c_prev'], cache['concat']
        
        # Output gate gradient
        tanh_c = np.tanh(c)
        do = dh * tanh_c
        do_raw = do * o * (1 - o)
        
        # Cell state gradient
        dc = dh * o * (1 - tanh_c**2) + dc_next
        
        # Forget gate gradient
        df = dc * c_prev
        df_raw = df * f * (1 - f)
        
        # Input gate gradient
        di = dc * c_hat
        di_raw = di * i * (1 - i)
        
        # Candidate gradient
        dc_hat = dc * i
        dc_hat_raw = dc_hat * (1 - c_hat**2)
        
        # Weight gradients (clip for stability)
        clip_val = 5.0
        
        for W, d_raw, b in [
            (self.W_f, df_raw, self.b_f),
            (self.W_i, di_raw, self.b_i),
            (self.W_c, dc_hat_raw, self.b_c),
            (self.W_o, do_raw, self.b_o)
        ]:
            dW = np.clip(np.outer(d_raw, concat), -clip_val, clip_val)
            db = np.clip(d_raw, -clip_val, clip_val)
            W -= lr * dW
            b -= lr * db
        
        # Previous hidden and cell gradients
        dc_prev = dc * f
        
        return dc_prev


# ============================================================
# LSTM AUTOENCODER
# ============================================================
class LSTMAutoencoder:
    """
    LSTM Autoencoder: Encoder LSTM + Dense Decoder
    
    Mimari:
    Input (seq_len, features) -> LSTM Encoder -> Latent -> Dense Decoder -> Reconstruction
    
    Normal verilerle egitilir.
    Reconstruction error yuksek olan ornek = anomali.
    """
    def __init__(self, input_size, hidden_size=32, latent_size=16):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Encoder LSTM
        self.encoder = LSTMCell(input_size, hidden_size)
        
        # Latent projection
        scale = np.sqrt(2.0 / (hidden_size + latent_size))
        self.W_latent = np.random.randn(latent_size, hidden_size) * scale
        self.b_latent = np.zeros(latent_size)
        
        # Decoder (dense layers)
        scale2 = np.sqrt(2.0 / (latent_size + hidden_size))
        self.W_dec1 = np.random.randn(hidden_size, latent_size) * scale2
        self.b_dec1 = np.zeros(hidden_size)
        
        scale3 = np.sqrt(2.0 / (hidden_size + input_size))
        self.W_out = np.random.randn(input_size, hidden_size) * scale3
        self.b_out = np.zeros(input_size)
    
    def encode(self, sequence):
        """Sekans -> Latent vektor"""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        caches = []
        
        for t in range(len(sequence)):
            h, c, cache = self.encoder.forward(sequence[t], h, c)
            caches.append(cache)
        
        # Latent projection
        latent = np.tanh(self.W_latent @ h + self.b_latent)
        return latent, h, caches
    
    def decode(self, latent, seq_len):
        """Latent vektor -> Reconstruction"""
        # Dense decoder
        hidden = np.tanh(self.W_dec1 @ latent + self.b_dec1)
        
        # Ayni cikiyi her adim icin kullan (basit decoder)
        output_step = self.W_out @ hidden + self.b_out
        reconstruction = np.tile(output_step, (seq_len, 1))
        
        return reconstruction
    
    def forward(self, sequence):
        """Tam forward pass"""
        latent, h, caches = self.encode(sequence)
        reconstruction = self.decode(latent, len(sequence))
        return reconstruction, latent, caches
    
    def compute_loss(self, original, reconstruction):
        """MSE Loss"""
        return np.mean((original - reconstruction) ** 2)
    
    def train_step(self, sequence, lr=0.001):
        """Tek egitim adimi"""
        reconstruction, latent, caches = self.forward(sequence)
        loss = self.compute_loss(sequence, reconstruction)
        
        # Simplified gradient (decoder)
        error = reconstruction - sequence
        mean_error = np.mean(error, axis=0)
        
        # Decoder gradients
        hidden = np.tanh(self.W_dec1 @ latent + self.b_dec1)
        
        # Output layer
        d_hidden = self.W_out.T @ mean_error
        self.W_out -= lr * np.clip(np.outer(mean_error, hidden), -5, 5)
        self.b_out -= lr * np.clip(mean_error, -5, 5)
        
        # Hidden layer
        d_hidden *= (1 - hidden**2)
        d_latent = self.W_dec1.T @ d_hidden
        self.W_dec1 -= lr * np.clip(np.outer(d_hidden, latent), -5, 5)
        self.b_dec1 -= lr * np.clip(d_hidden, -5, 5)
        
        # Latent layer
        d_latent *= (1 - latent**2)
        h = caches[-1]['h']
        self.W_latent -= lr * np.clip(np.outer(d_latent, h), -5, 5)
        self.b_latent -= lr * np.clip(d_latent, -5, 5)
        
        # Encoder BPTT (son 5 adim)
        dh = self.W_latent.T @ d_latent
        dc = np.zeros(self.hidden_size)
        bptt_steps = min(5, len(caches))
        for t in range(len(caches)-1, max(len(caches)-bptt_steps-1, -1), -1):
            dc = self.encoder.backward(dh, dc, caches[t], lr=lr)
            if t > 0:
                dh = np.zeros(self.hidden_size)  # Simplified
        
        return loss
    
    def fit(self, X_train, epochs=10, lr=0.001, batch_print=5):
        """Model egitimi"""
        losses = []
        n = len(X_train)
        
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(n)
            
            for idx in indices:
                loss = self.train_step(X_train[idx], lr=lr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n
            losses.append(avg_loss)
            
            # Learning rate decay
            if epoch > 0 and epoch % 10 == 0:
                lr *= 0.8
            
            if (epoch + 1) % batch_print == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f}")
        
        return losses
    
    def get_anomaly_scores(self, X):
        """Her ornek icin reconstruction error hesapla"""
        scores = []
        for seq in X:
            recon, _, _ = self.forward(seq)
            score = np.mean((seq - recon) ** 2)
            scores.append(score)
        return np.array(scores)


# ============================================================
# VERI HAZIRLAMA
# ============================================================
def prepare_sequences(raw_path, seq_length=48):
    """
    Ham tuketim verisini LSTM icin sekans formatina cevir.
    seq_length=48: 1 gunluk (15dk x 96 = 24 saat)
    """
    print("[1/3] Veriler hazirlaniyor...")
    
    raw = pd.read_csv(raw_path)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    
    # Features CSV'den etiketleri al
    feat_path = raw_path.replace('raw_consumption_sample.csv', 'features.csv')
    labels_df = pd.read_csv(feat_path)[['customer_id', 'label', 'theft_type']]
    
    scaler = MinMaxScaler()
    
    sequences = []
    labels = []
    customer_ids = []
    
    for cid in raw['customer_id'].unique():
        cust_data = raw[raw['customer_id'] == cid].sort_values('timestamp')
        consumption = cust_data['consumption_kw'].values.reshape(-1, 1)
        
        # Normalize
        consumption_scaled = scaler.fit_transform(consumption)
        
        # Sekanslara bol (her gun 1 sekans)
        n_sequences = len(consumption_scaled) // seq_length
        for i in range(min(n_sequences, 5)):  # Max 30 gun (hiz icin)
            seq = consumption_scaled[i*seq_length:(i+1)*seq_length]
            if len(seq) == seq_length:
                sequences.append(seq)
                label = labels_df[labels_df['customer_id'] == cid]['label'].values[0]
                labels.append(label)
                customer_ids.append(cid)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"    Sekans sayisi: {len(X)}")
    print(f"    Sekans boyutu: {X.shape[1]} adim x {X.shape[2]} ozellik")
    print(f"    Normal: {(y==0).sum()}, Kacak: {(y==1).sum()}")
    
    return X, y, np.array(customer_ids), scaler


# ============================================================
# ANA FONKSIYON
# ============================================================
def main():
    print("=" * 60)
    print("MASS-AI: LSTM Autoencoder — Sifirdan Implementasyon")
    print("=" * 60)
    start_time = time.time()
    
    # Veri hazirla
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'raw_consumption_sample.csv')
    X, y, cids, scaler = prepare_sequences(data_path, seq_length=48)
    
    # Train/test ayir (sadece normal veri ile egit)
    normal_mask = y == 0
    X_normal = X[normal_mask]
    
    n_train = int(len(X_normal) * 0.8)
    X_train = X_normal[:n_train]
    X_val = X_normal[n_train:]
    
    print(f"\n    Egitim (normal): {len(X_train)}")
    print(f"    Validasyon (normal): {len(X_val)}")
    print(f"    Test (tumu): {len(X)}")
    
    # Model olustur ve egit
    print("\n[2/3] LSTM Autoencoder egitiliyor...")
    print(f"    Mimari: Input(48x1) -> LSTM(32) -> Latent(16) -> Dense -> Output(96x1)")
    
    model = LSTMAutoencoder(input_size=1, hidden_size=32, latent_size=16)
    losses = model.fit(X_train, epochs=10, lr=0.002, batch_print=5)
    
    # Anomali skorlari
    print("\n[3/3] Anomali skorlari hesaplaniyor...")
    all_scores = model.get_anomaly_scores(X)
    
    # Esik belirleme (validation seti uzerinden)
    val_scores = model.get_anomaly_scores(X_val)
    threshold = np.percentile(val_scores, 95)  # %95 percentil
    print(f"    Esik degeri (95. percentil): {threshold:.6f}")
    
    # Tahminler
    y_pred = (all_scores > threshold).astype(int)
    
    # Sonuclar
    print("\n" + "=" * 60)
    print("LSTM AUTOENCODER SONUCLARI")
    print("=" * 60)
    
    print("\nSiniflandirma Raporu:")
    print(classification_report(y, y_pred, target_names=['Normal', 'Kacak']))
    
    auc = roc_auc_score(y, all_scores)
    f1 = f1_score(y, y_pred)
    print(f"ROC-AUC Skoru: {auc:.4f}")
    print(f"F1 Skoru (Kacak): {f1:.4f}")
    
    # Musteri bazli analiz
    print("\nMusteri Bazli Analiz:")
    cust_scores = {}
    for cid, score, label in zip(cids, all_scores, y):
        if cid not in cust_scores:
            cust_scores[cid] = {'scores': [], 'label': label}
        cust_scores[cid]['scores'].append(score)
    
    cust_avg_scores = []
    cust_labels = []
    for cid, data in cust_scores.items():
        cust_avg_scores.append(np.mean(data['scores']))
        cust_labels.append(data['label'])
    
    cust_avg_scores = np.array(cust_avg_scores)
    cust_labels = np.array(cust_labels)
    
    cust_threshold = np.percentile(cust_avg_scores[cust_labels == 0], 95)
    cust_preds = (cust_avg_scores > cust_threshold).astype(int)
    
    cust_auc = roc_auc_score(cust_labels, cust_avg_scores)
    cust_f1 = f1_score(cust_labels, cust_preds)
    print(f"Musteri Bazli ROC-AUC: {cust_auc:.4f}")
    print(f"Musteri Bazli F1: {cust_f1:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nToplam sure: {elapsed:.1f} saniye")
    
    # ===== GRAFIKLER =====
    print("\nGrafikler olusturuluyor...")
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MASS-AI: LSTM Autoencoder Sonuclari', fontsize=16, fontweight='bold', color='#1B4F72')
    
    # 1. Training loss
    ax = axes[0, 0]
    ax.plot(losses, color='#1B4F72', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Egitim Kaybi (Training Loss)')
    ax.grid(True, alpha=0.3)
    
    # 2. Anomali skor dagilimi
    ax = axes[0, 1]
    normal_scores = all_scores[y == 0]
    theft_scores = all_scores[y == 1]
    ax.hist(normal_scores, bins=40, alpha=0.7, color='#2E86C1', label=f'Normal (n={len(normal_scores)})', density=True)
    ax.hist(theft_scores, bins=40, alpha=0.7, color='#E74C3C', label=f'Kacak (n={len(theft_scores)})', density=True)
    ax.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Esik={threshold:.4f}')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Yogunluk')
    ax.set_title('Anomali Skor Dagilimi')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    ax = axes[1, 0]
    precision, recall, _ = precision_recall_curve(y, all_scores)
    ax.plot(recall, precision, color='#1B4F72', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.1, color='#2E86C1')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (AUC={auc:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Reconstruction ornekleri
    ax = axes[1, 1]
    # Normal ornek
    normal_idx = np.where(y == 0)[0][0]
    recon_normal, _, _ = model.forward(X[normal_idx])
    ax.plot(X[normal_idx][:, 0], color='#2E86C1', linewidth=1, alpha=0.7, label='Orijinal (Normal)')
    ax.plot(recon_normal[:, 0], color='#2E86C1', linewidth=1, linestyle='--', alpha=0.7, label='Reconstruction')
    
    # Kacak ornek
    theft_idx = np.where(y == 1)[0][0]
    recon_theft, _, _ = model.forward(X[theft_idx])
    ax.plot(X[theft_idx][:, 0], color='#E74C3C', linewidth=1, alpha=0.7, label='Orijinal (Kacak)')
    ax.plot(recon_theft[:, 0], color='#E74C3C', linewidth=1, linestyle='--', alpha=0.7, label='Reconstruction')
    
    ax.set_xlabel('Zaman Adimi (15dk)')
    ax.set_ylabel('Normalize Tuketim')
    ax.set_title('Reconstruction Ornekleri')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(docs_dir, 'lstm_autoencoder_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {fig_path}")
    
    # ===== KARSILASTIRMA TABLOSU =====
    print("\n" + "=" * 60)
    print("TUM MODELLER — KARSILASTIRMA")
    print("=" * 60)
    print(f"{'Model':<25} {'Tip':<15} {'ROC-AUC':>10} {'F1':>8}")
    print("-" * 60)
    print(f"{'Isolation Forest':<25} {'Unsupervised':<15} {'0.8208':>10} {'0.2609':>8}")
    print(f"{'Random Forest':<25} {'Supervised':<15} {'0.9471':>10} {'0.8704':>8}")
    print(f"{'XGBoost':<25} {'Supervised':<15} {'0.9373':>10} {'0.8468':>8}")
    print(f"{'LSTM Autoencoder':<25} {'Deep Learning':<15} {auc:>10.4f} {f1:>8.4f}")
    print("=" * 60)
    print("\nNOT: LSTM Autoencoder unsupervised calisir — etiketsiz veriyle")
    print("kacak tespiti yapabilmesi en buyuk avantaji.")


if __name__ == "__main__":
    main()
