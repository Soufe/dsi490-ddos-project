# ข้อกำหนดและกรอบแนวทางโครงการ (Project Specification & Ground Truth)
> **สถานะ**: ห้ามแก้ไขเอกสารนี้โดยพลการ (Strictly Read-Only / Ground Truth)  
> **ที่มา**: Google Docs (DSI490 Project - DDoS Detection & Mitigation on SDN using DNN)  
> **URL อ้างอิง**: https://docs.google.com/document/d/1dnEw2MMW00vsTVLC6u63MPPqgj9GERItPxVWdbGlZdU/edit?usp=sharing

---

## 1. ชื่อโครงการและข้อมูลทั่วไป
- **ชื่อภาษาไทย**: ระบบตรวจจับและป้องกันการโจมตี DDoS แบบอัตโนมัติบนเครือข่าย Software-Defined Network โดยใช้ Deep Neural Network
- **ชื่อภาษาอังกฤษ**: Automatic DDoS Detection and Mitigation System on SDN using Deep Neural Network
- **วิชา**: วข.490 / DSI490 (Senior Project / Capstone Project, Data Science & Innovation, มหาวิทยาลัยธรรมศาสตร์)
- **อาจารย์ที่ปรึกษา**: อ.สมเกียรติ โกศลสมบัติ (Somkiat Kosolsombat)
- **สมาชิกกลุ่ม**: เฟิร์ส & เฟิน (Soufe)
- **GitHub Repository**: https://github.com/Soufe/dsi490-ddos-project

---

## 2. วัตถุประสงค์หลัก (Core Objectives)
1. พัฒนาโมเดล Deep Neural Network (DNN) สำหรับตรวจจับ Network Traffic ที่เป็นการโจมตี DDoS บนเครือข่าย SDN
2. พัฒนาระบบอัตโนมัติให้ SDN Controller สามารถออกกฎ (Flow Rule / FlowMod) บน OpenFlow Switch เพื่อบล็อก/ทิ้ง (Drop) หรือจำกัดการโจมตีได้แบบอัตโนมัติเมื่อตรวจพบ
3. ประเมินประสิทธิภาพระบบแบบรอบด้าน ทั้งด้าน Detection Accuracy (ความแม่นยำในการตรวจจับ) และ Practical Mitigation Latency / Response Time รวมถึงผลกระทบต่อ Legitimate Traffic

---

## 3. สถาปัตยกรรมและขั้นตอนการทำงานของระบบ (System Architecture & Workflow)

```text
[Mininet: จำลอง Network Topology + Host ปกติ + Attacker (hping3/Scapy)]
                          │
                          ▼
[Ryu SDN Controller] ◄──OpenFlow──► [OpenFlow Switch (Data Plane)]
        │                                        ▲
        ▼ (OFPFlowStatsRequest / Reply ทุก N วินาที)│
[Feature Extraction & Scaler (scaler.pkl)]        │
        │                                        │
        ▼                                        │
[DNN Model Inference (Pre-trained)]              │
        │                                        │
        ▼                                        │
[Adaptive Threshold & Suspicion Score]           │
        │                                        │
        ├─► Suspicion Score >= THRESHOLD         │
        │   └─► ออก Flow Rule (FlowMod Drop, hard_timeout=300s) ──┘
        │   └─► บันทึกลง Log & Real-time Dashboard
        │
        └─► Suspicion Score < THRESHOLD
            └─► ปล่อย Traffic ตามปกติ (Decay Score เมื่อเป็น Benign)
```

### รายละเอียด 4 โมดูลหลัก (4 Core Modules):
1. **Module 1 - Data Collection & Preparation**:
   - ใช้ชุดข้อมูล CIC-DDoS2019 (Day 1: `CSV-01-12`, Day 2: `CSV-03-11`) และ InSDN
   - ประมวลผลผ่าน `data_prep.py`: Chunking (100k แถว), กำจัด Metadata/Leakage Columns (IP, Port, Timestamp, Flow ID), ทำความสะอาด Inf/NaN, จัดการ Class Imbalance (Undersampling/Balancing), สกัด Feature ที่สอดคล้องกับ OpenFlow
   - บันทึก `scaler.pkl`, `feature_columns.json`, `label_mapping.json`
2. **Module 2 - DNN Model**:
   - โครงสร้าง Feedforward DNN 3–5 Hidden Layers (เช่น 128-256-128 units พร้อม Dropout & Batch Normalization)
   - ป้องกัน Overfitting ด้วย Regularization/Dropout, Activation: ReLU ใน Hidden layers, Softmax/Sigmoid ใน Output layer
   - จัดการปัญหา Class Imbalance ด้วย Class Weight / Focal Loss / SMOTE
3. **Module 3 - Detection Integration (Ryu Application)**:
   - Ryu Application (Python) ส่ง `OFPFlowStatsRequest` ดึง Flow Statistics เป็นระยะ
   - ทำ Feature Engineering จาก Flow Stats (Packet rate, Byte rate, Duration, Packet size mean ฯลฯ)
   - Scale ด้วย `scaler.pkl` และส่งเข้า DNN Model เพื่อ Inference แบบ Real-time
4. **Module 4 - Mitigation & False Positive Management**:
   - **Adaptive Threshold / Suspicion Score**: มี Confirmation Window ไม่สั่งบล็อกจากการ Predict ผิดปกติเพียงครั้งเดียว (Attack: +1, Benign: -1 decay)
   - **Dynamic Flow Rule**: เมื่อคะแนนถึงเกณฑ์ ส่ง `OFPFlowMod` (Match Source IP, Action Drop, Priority สูง) ให้ Switch จัดการที่ Data Plane โดยตรง
   - **Auto-Unblock Recovery**: กำหนด `hard_timeout` (เช่น 300 วินาที) ให้ Switch ลบ Flow Rule อัตโนมัติ เพื่อคืนสิทธิ์ให้ Host ปกติในกรณี False Positive

---

## 4. ช่องว่างงานวิจัยที่โปรเจคนี้เน้นต่อยอด (Research Gaps)
1. **End-to-End Closed-Loop System**: ไม่ได้หยุดแค่วัดความแม่นยำของ Model แต่เชื่อมโยงครบวงจรตั้งแต่ Traffic Ingestion -> DNN -> Flow Rule Enforcement
2. **False Positive Management**: มีกลไก Suspicion Score / Confirmation Window ลดการบล็อกทราฟฟิกปกติผิดพลาด
3. **Auto-Unblock Mechanism**: แก้ปัญหาการบล็อกค้างถาวรด้วย Hard Timeout
4. **Practical Metric Evaluation**: ประเมิน Detection Latency, Mitigation Response Time (ms/s), Controller CPU Utilization และ Throughput ของ Legitimate Traffic ภายหลังการบรรเทาภัย

---

## 5. มาตรวัดประสิทธิภาพ (Evaluation Metrics)
- **Detection Metrics**: Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), ROC-AUC
- **Mitigation & System Metrics**: Detection Time (ms), Mitigation Response Time (ms), Controller CPU & Memory Load, Packet Loss & Throughput ของทราฟฟิกปกติ (BENIGN)

---

## 6. ข้อกำหนดในการพัฒนาโค้ด (Implementation Rules)
- โค้ด Data Prep และ Training ต้องใช้ Artifacts ร่วมกับ Ryu Controller (`scaler.pkl`, `feature_columns.json`, `label_mapping.json`) อย่างสอดคล้องกัน ป้องกัน Data Leakage และ Feature Mismatch
- ตรวจสอบว่า Feature ที่ใช้ใน DNN เป็นค่าที่ OpenFlow Flow Statistics สามารถสกัดได้จริงในสภาพแวดล้อม Ryu + Mininet
- ยึดแนวทางตามโครงสร้างรายงานและ Timeline ของหลักสูตร DSI490
