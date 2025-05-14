import os
import numpy as np
import cv2
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, padding
from scipy.stats import entropy as calc_entropy  # Added for entropy calculation

# ========================== BB84 QKD with Eavesdrop Detection ==========================

def generate_bb84_key(num_qubits=20, eavesdrop=False):
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be greater than 0.")

    alice_basis = np.random.choice(['Z', 'X'], size=num_qubits)
    alice_state = np.random.randint(2, size=num_qubits)
    bob_basis = np.random.choice(['Z', 'X'], size=num_qubits)

    print("\n--- BB84 with Eavesdropping Simulation ---")
    print("Alice's Bits:    ", ' '.join(map(str, alice_state)))
    print("Alice's Bases:   ", ' '.join(alice_basis))
    print("Bob's Bases:     ", ' '.join(bob_basis))

    eve_basis = np.random.choice(['Z', 'X'], size=num_qubits)
    eve_measured_bits = []

    for i in range(num_qubits):
        if eavesdrop:
            if eve_basis[i] == alice_basis[i]:
                eve_measured_bits.append(alice_state[i])
            else:
                eve_measured_bits.append(np.random.randint(2))
        else:
            eve_measured_bits = alice_state.copy()
            break

    received_state = []
    for i in range(num_qubits):
        received_state.append(eve_measured_bits[i])

    bob_measurements = []
    for i in range(num_qubits):
        if bob_basis[i] == alice_basis[i]:
            bob_measurements.append(received_state[i])
        else:
            bob_measurements.append(np.random.randint(2))

    key_alice = []
    key_bob = []
    for i in range(num_qubits):
        if alice_basis[i] == bob_basis[i]:
            key_alice.append(str(alice_state[i]))
            key_bob.append(str(bob_measurements[i]))

    print("Sifted Alice Key:", ''.join(key_alice))
    print("Sifted Bob Key:  ", ''.join(key_bob))

    mismatches = sum(1 for a, b in zip(key_alice, key_bob) if a != b)
    total = len(key_alice)
    error_rate = mismatches / total if total > 0 else 0
    print(f"Error Rate: {error_rate:.2%}")

    if error_rate > 0.3:
        print("⚠️  Possible eavesdropping detected! Aborting key usage.")
        return None
    else:
        print("✅  Key exchange successful.")

        # 🔐 Entropy calculation of the actual BB84 key
        def calculate_entropy_from_bitstring(bitstring):
            counts = np.bincount([int(b) for b in bitstring], minlength=2)
            probs = counts / len(bitstring)
            return calc_entropy(probs, base=2)

        entropy_val = calculate_entropy_from_bitstring(key_alice)
        print(f"Key Entropy: {entropy_val:.4f} bits per bit")

        final_key_bytes = bytes(int(b) for b in key_alice[:16])
        shared_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"bb84-key",
        ).derive(final_key_bytes)
        return shared_key

# ========================== AES Encryption and Decryption ==========================

class AESCipher:
    def __init__(self, key):
        self.key = key

    def encrypt(self, data):
        iv = os.urandom(16)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        return iv + encryptor.update(padded_data) + encryptor.finalize()

    def decrypt(self, enc_data):
        iv = enc_data[:16]
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(enc_data[16:]) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(decrypted_padded_data) + unpadder.finalize()

# ========================== Video Processing ==========================

def extract_frames(video_path, frame_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    os.makedirs(frame_folder, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_folder, f"frame_{frame_count}.png"), frame)
        frame_count += 1
    cap.release()
    return frame_count, fps

def encrypt_frames(input_folder, enc_folder, enc_img_folder, aes_cipher):
    os.makedirs(enc_folder, exist_ok=True)
    os.makedirs(enc_img_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        img_data = image.tobytes()
        encrypted_data = aes_cipher.encrypt(img_data)
        with open(os.path.join(enc_folder, filename + ".enc"), "wb") as f:
            f.write(encrypted_data)
        encrypted_image = np.frombuffer(encrypted_data[:len(img_data)], dtype=np.uint8).reshape(image.shape)
        cv2.imwrite(os.path.join(enc_img_folder, filename.replace(".png", ".jpg")), encrypted_image)
    print("Encryption completed.")

def decrypt_frames(input_folder, output_folder, aes_cipher, frame_shape):
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith(".enc"):
            continue
        with open(os.path.join(input_folder, filename), "rb") as f:
            encrypted_data = f.read()
        decrypted_data = aes_cipher.decrypt(encrypted_data)
        expected_size = np.prod(frame_shape)
        if len(decrypted_data) < expected_size:
            print(f"Warning: Decrypted data size mismatch for {filename}. Skipping...")
            continue
        image = np.frombuffer(decrypted_data[:expected_size], dtype=np.uint8).reshape(frame_shape)
        cv2.imwrite(os.path.join(output_folder, filename.replace(".enc", ".png")), image)
    print("Decryption completed.")

def create_video_from_frames(input_folder, output_video, fps):
    frame_files = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".png")],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    if not frame_files:
        print("No frames found!")
        return
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    print(f"Video reconstruction completed: {output_video}")

# ========================== Video Processing Wrapper ==========================

def process_video(video_path, aes_cipher):
    frame_folder = "frames"
    encrypted_folder = "encrypted_frames"
    encrypted_images_folder = "encrypted_images_jpg"
    decrypted_folder = "decrypted_frames"
    output_video = "output.mp4"
    os.makedirs(encrypted_images_folder, exist_ok=True)
    num_frames, original_fps = extract_frames(video_path, frame_folder)
    print(f"Extracted {num_frames} frames at {original_fps} FPS.")
    encrypt_frames(frame_folder, encrypted_folder, encrypted_images_folder, aes_cipher)
    first_frame = cv2.imread(os.path.join(frame_folder, "frame_0.png"))
    if first_frame is None:
        print("Error: Could not read the first frame.")
        return
    frame_shape = first_frame.shape
    decrypt_frames(encrypted_folder, decrypted_folder, aes_cipher, frame_shape)
    create_video_from_frames(decrypted_folder, output_video, original_fps)

# ========================== Message Encryption ==========================

def process_message(aes_cipher):
    message = input("Enter the message to encrypt: ").encode()
    encrypted_msg = aes_cipher.encrypt(message)
    decrypted_msg = aes_cipher.decrypt(encrypted_msg).decode()
    print("Encrypted Message:", encrypted_msg.hex())
    print("Decrypted Message:", decrypted_msg)

# ========================== Main Menu ==========================

if __name__ == "__main__":
    print("Select an option:")
    print("1. Send a message")
    print("2. Send a video")
    choice = input("Enter choice (1/2): ")
    shared_key = generate_bb84_key(num_qubits=20, eavesdrop=True)
    if shared_key is None:
        print("Key exchange failed due to possible eavesdropping. Exiting...")
        exit()
    aes_cipher = AESCipher(shared_key)
    if choice == "1":
        process_message(aes_cipher)
    elif choice == "2":
        video_path = input("Enter video file path: ")
        process_video(video_path, aes_cipher)
    else:
        print("Invalid choice!")
