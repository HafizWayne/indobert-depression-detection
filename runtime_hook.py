import os
import sys

# Mengambil direktori tempat .exe berada
exe_dir = os.path.dirname(sys.executable)

# Membuat path untuk subfolder baru bernama '_internal'
temp_dir = os.path.join(exe_dir, '_internal')

# Membuat subfolder tersebut jika belum ada
if not os.path.exists(temp_dir):
    try:
        os.makedirs(temp_dir)
    except OSError:
        # Menangani kasus jika folder tidak bisa dibuat (misal: karena izin)
        # Fallback ke folder temp sistem jika gagal
        pass

# Mengarahkan ekstraksi ke subfolder baru
os.environ['MEIPASS2'] = temp_dir