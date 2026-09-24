# Smart India Hackathon (SIH) Presentation Content
*Forensic Face Photo-Sketch Recognition System*

**Instructions for Use:** 
Copy and paste the contents of each "Slide" below into your PowerPoint presentation. Use bullet points and keep text minimal on the actual slides; use the "Speaker Notes" section to prepare what you will actually say during the pitch.

---

## SLIDE 1: Title Slide
**Title:** Forensic Face Photo-Sketch Recognition Using AI
**Subtitle:** Bridging the Modality Gap for Law Enforcement
**Problem Statement ID:** [Insert Your SIH Problem ID Here]
**Team Name:** [Insert Team Name]
**Theme:** Security & Surveillance / Smart Automation

**Speaker Notes:**
*Good morning judges. We are Team [Name], and we are tackling Problem Statement [ID]. Our project aims to revolutionize how law enforcement matches forensic police sketches to mugshot databases using Artificial Intelligence.*

---

## SLIDE 2: The Problem We Are Solving
**Title: The Forensic Matching Challenge**
- **The Modality Gap:** Sketches lack texture, color, and depth. Standard face recognition algorithms (designed for photos) completely fail on hand-drawn sketches.
- **Subjective Distortion:** "Forensic" sketches drawn from human memory often contain severe geometric errors (e.g., eyes too wide, jaw too sharp).
- **The Current Process:** Law enforcement relies on slow, manual human comparison or highly inaccurate traditional software.
- **GAN Hallucination Risk:** Generating a "fake photo" from a sketch using Generative AI (GANs) often changes the suspect's identity, which is unacceptable in legal forensics.

**Speaker Notes:**
*When a crime occurs without CCTV footage, police rely on composite sketches. However, you cannot just plug a sketch into a standard facial recognition system—it will fail. The gap between a textured photo and a sparse sketch is too wide, and existing AI often 'hallucinates' fake features that can wrongly accuse someone.*

---

## SLIDE 3: Our Proposed Solution
**Title: Cross-Modal Alignment & Score-Level Fusion**
- **Deep Transfer Learning:** A neural network (InceptionResnetV1) fine-tuned specifically to map both sketches and photos into the same mathematical space.
- **Dual-Stage Metric Learning:** Uses **Triplet Margin Loss** to push matching photo-sketch pairs closer while pushing distractors away.
- **Test-Time Augmentation:** Mathematically simulates multiple variations of the sketch to compensate for artist drawing errors.
- **System Fusion:** Combines modern AI semantics with a traditional **HOG (Histogram of Oriented Gradients)** edge-detector as a safety net against AI overfitting.

**Speaker Notes:**
*Our solution completely bypasses generating fake photos. Instead, we use metric learning to translate both the photo and the sketch into a shared mathematical language. We also simulate different structural variations of the sketch at test time to account for drawing errors, and fuse the AI's decision with a traditional edge-detection algorithm to ensure maximum accuracy.*

---

## SLIDE 4: Technical Architecture
**Title: How It Works (System Architecture)**

*[Visual Suggestion: Create a flow chart based on these steps]*
1. **Input:** Forensic Sketch
2. **Augmentation Layer:** Rapid 2D/3DMM Morphological variants generated.
3. **Branch 1 (AI):** Fine-Tuned Deep Neural Network $\rightarrow$ Extracts 512-D Semantic Vector.
4. **Branch 2 (Classical):** HOG Feature Extractor $\rightarrow$ Extracts Edge Structures.
5. **Fusion Layer:** Min-Max Normalization + Sum of Scores (L2 Distance + Spearman Correlation).
6. **Output:** Ranked List of Mugshots (Rank-1 to Rank-10 matches).

**Speaker Notes:**
*Our architecture is a dual-branch system. The input sketch is augmented and fed into both a deep neural network and a traditional HOG edge-extractor. The deep network understands the global semantics of the face, while HOG ensures the geometric structure matches. We fuse these two independent scores to rank the mugshot database, outputting a shortlist of suspects.*

---

## SLIDE 5: The Showstopper (Innovation)
**Title: What Makes Our Solution Unique?**
- **No Identity Hallucination:** Unlike GANs, Triplet Loss metric learning preserves the true geometry of the suspect.
- **Heterogeneous Score-Level Fusion:** We don't blindly trust AI. We use Spearman Rank Correlation on HOG features to independently verify the neural network's guess.
- **Test-Time Morphological Perturbation:** The system doesn't just evaluate the sketch; it evaluates *K-variants* of the sketch, actively guessing what the suspect might look like under different artist assumptions.
- **Proven Scalability:** Tested against massive unconstrained distractor galleries (LFW) to prove it works in real-world database dilutions.

**Speaker Notes:**
*The innovation here is that we don't blindly trust deep learning, because deep learning overfits on small sketch datasets. By fusing AI with Spearman rank correlation—which compares the rank-order of edges rather than pixel intensity—our system is incredibly robust to the severe intensity changes between photos and sketches.*

---

## SLIDE 6: Technology Stack
**Title: Tech Stack & Tools Used**
- **Core Language:** Python 3
- **AI & Deep Learning:** PyTorch, torchvision, facenet-pytorch
- **Computer Vision:** scikit-image, Pillow, Matplotlib
- **Data & Math:** scikit-learn, SciPy, NumPy
- **Model Backbone:** InceptionResnetV1 (Pretrained on VGGFace2)
- **3D Modeling (Offline):** Basel Face Model 2019 (HDF5)

**Speaker Notes:**
*We built this entirely in Python using PyTorch for dynamic computation. We utilized Transfer Learning via a model pre-trained on VGGFace2, saving immense computational resources, and engineered a custom headless 3D software renderer to simulate facial variance.*

---

## SLIDE 7: Use Cases & Feasibility
**Title: Target Audience & Practical Deployment**
- **Primary Users:** Police Departments, Forensic Investigators, Intelligence Agencies.
- **Feasibility:** 
  - Extremely lightweight inference (takes milliseconds per query on a GPU).
  - Can be wrapped into a REST API microservice and integrated into existing police database software (e.g., CCTNS in India).
  - Works offline; highly secure and preserves data privacy.
- **Cost Effectiveness:** Open-source stack. Requires no proprietary software licenses.

**Speaker Notes:**
*This system is highly feasible for immediate deployment. Because inference is lightweight, it can be hosted on standard police servers without needing a massive supercomputer. It can plug directly into existing databases like CCTNS, providing investigators with an instant suspect shortlist.*

---

## SLIDE 8: Future Scope
**Title: Scaling Up**
- **Automated Face Alignment:** Integrating MTCNN/RetinaFace to automatically crop unaligned CCTV frames and database photos.
- **Differentiable 3D Rendering:** Upgrading the CPU-based 3DMM simulator to PyTorch3D for GPU-accelerated, end-to-end training.
- **Multi-Modal Text Input:** Expanding the latent space to accept natural language witness descriptions (e.g., "tall man with a scar") alongside the visual sketch.

**Speaker Notes:**
*In the future, we plan to fully automate the preprocessing pipeline so it can ingest raw, unaligned photos. We also aim to upgrade our 3D simulator to run on the GPU, and eventually incorporate NLP so investigators can search using both the sketch and the witness's verbal description simultaneously.*

---

## SLIDE 9: Conclusion
**Title: Thank You**
- **Team Name:** [Insert Team Name]
- **Contact:** [Insert Email/Phone]
- **Q&A**

**Speaker Notes:**
*Thank you for your time. We are now open to any questions you have regarding the architecture, algorithms, or implementation.*
