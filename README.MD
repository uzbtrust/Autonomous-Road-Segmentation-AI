# 🏎️ Autonomous Road Segmentation AI: Tesla-Level Perception

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00ff80&height=150&section=header&text=Autonomous%20Vision&fontSize=50" width="100%"/>
</div>

## 🌟 Overview
This project showcases a high-speed **Semantic Segmentation** system built for autonomous driving. Optimized for high-frequency inference, it achieves ultra-low latency, making it ideal for real-time safety systems.

---

## 🛠️ Visual Journey & Performance

<table align="center">
  <tr>
    <td align="center">
      <h3>1. Model Training & Concept</h3>
      <p>Initial validation results showing original scenes, ground truth, and AI predictions.</p>
      <img src="image_cd6b61.jpg" width="600" alt="Model Training Results"/>
      <br><i>Accuracy: 94.2% | Dataset: CamVid</i>
    </td>
  </tr>

  <tr>
    <td align="center">
      <br>
      <h3>2. Static Image Inference</h3>
      <p>Performance benchmark on high-resolution road scenes.</p>
      <div style="border: 2px solid #00ff80; border-radius: 10px; padding: 10px;">
        <img src="image_cd6ee3.jpg" width="600" alt="Image Inference Performance"/>
        <br><b>⚡ M4 Inference: 1.65 ms</b>
      </div>
    </td>
  </tr>

  <tr>
    <td align="center">
      <br>
      <h3>3. Real-Time Video Power</h3>
      <p>The system's peak performance during live video stream processing.</p>
      <div style="background: #101010; border-radius: 10px; padding: 20px;">
        <img src="image_cd63a1.jpg" width="500" alt="Video FPS Performance"/>
        <br><h2 style="color: #00ff80;">🚀 1163.3 FPS | 0.9ms Latency</h2>
      </div>
    </td>
  </tr>
</table>

---

## 🖥️ Interactive Dashboard
Below is the live preview of the autonomous vision control center. It allows for seamless image and video processing.

<p align="center">
  <img src="image_6ea1.jpg" width="900" alt="Streamlit Website Preview"/>
</p>

---

## 📂 Project Structure
The repository is organized as follows to ensure scalability and ease of use:

<p align="center">
  <img src="image_6a61.jpg" width="400" alt="File Structure"/>
</p>

---

## 🚀 Technical Specifications
<p align="center">
  <img src="https://img.shields.io/badge/Model-U--Net-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-TensorFlow_2.15-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Backend-M4_Metal-blue?style=for-the-badge&logo=apple">
</p>

- **Real-time Processing:** Optimized pixel-wise classification.
- **Hardware Acceleration:** Native performance on Apple Silicon.
- **Interactive Dashboard:** Built-in support for image/video upload and webcam stream.

---

## 🏁 How to Run
1. **Clone:** `git clone https://github.com/uzbtrust/Autonomous-Road-Segmentation-AI.git`
2. **Install:** `pip install -r requirements.txt`
3. **Launch:** `streamlit run web.py`

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00ff80&height=100&section=footer" width="100%"/>
  <p>Developed with ❤️ by <b>uzbtrust</b></p>
</div>
