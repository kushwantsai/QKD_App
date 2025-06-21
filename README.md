
Quantum-Safe Chat & Video Encryption using BB84 and AES

This project implements a hybrid cryptographic system combining Quantum Key Distribution (BB84 protocol) with AES encryption to ensure secure message and video communication. The system can detect eavesdropping, calculate entropy of the generated quantum keys, and perform full video encryption and decryption.

What This Project Does

* Generates a quantum-safe key using the BB84 protocol (simulated with eavesdropping detection)
* Calculates and displays the entropy of the generated key
* Encrypts and decrypts messages or videos using AES-CBC
* Extracts and reconstructs video frames securely
* Saves and optionally visualizes BB84 key characteristics

How to Run the Project

1. Prerequisites

Ensure you have Python 3.9 or later and the following libraries installed:

```
pip install qiskit qiskit-aer cryptography opencv-python matplotlib scipy
```

2. Run the Program

To start the project, run the following command:

```
python BB84_main.py
```

You will be prompted:

```
Select an option:
1. Send a message
2. Send a video
```

Option 1: Encrypt a Message

* Enter any text message.
* The program will:

  * Generate a BB84 key
  * Display the entropy of the key
  * Encrypt and decrypt the message using AES

Option 2: Encrypt a Video

* Provide the full path of an .mp4 video file.
* make sure to include the extension while providing the path
* The program will:

  * Extract frames
  * Encrypt each frame
  * Decrypt and reconstruct the output video
  * Save the result as output.mp4

Educational Value

* Demonstrates core principles of quantum cryptography
* Reinforces secure communication using hybrid encryption
* Provides visual validation of quantum key entropy and security properties

This repository contains the official implementation of our research paper:

**"Quantum-Enhanced Secure Communication: Integrating BB84 Key Distribution with AES for Real-Time Text and Video Transmission"**,  
presented at the **16th IEEE ICCCNT 2025**, hosted by **IIT Indore**.

