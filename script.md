# 🎤 Presentation Script: Forensic Sketch-to-Photo Retrieval

This is a step-by-step narrative script for a live presentation or video demo. It guides you on **what to say**, **what to show on screen**, and **what commands to run** to give a "next-level" presentation that highlights both the technical depth and the visual results of your project.

---

## 🎬 Introduction: Setting the Stage (1 min)

**What to say:**
> "Welcome to the demo of our Forensic Sketch-to-Photo Retrieval system. The core challenge in forensics is matching a hand-drawn composite sketch to a real mugshot. Because sketches lack texture and have geometric distortions, traditional facial recognition fails. 
> 
> Today, I'll show you how we solve this using a multi-branch architecture: a deep learning embedding network trained on FaceNet, combined with hand-crafted features, and enhanced using 3D Morphable Models for data augmentation."

**Action:** 
- Open `DEMO.md` and scroll to the **System architecture** diagram to give the audience a visual map of what you are about to show.

---

## 🚀 Part 1: The Core Engine (2 mins)

**What to say:**
> "Let's start by seeing the core engine in action on a single sketch."

**Action:**
- Split your screen. On the left, open the image: `data/cufs/sketches/m-063-01-sz1.jpg`. 
- On the right, have your terminal ready.

**Command to run:**
```bash
python3 predict.py --sketch data/cufs/sketches/m-063-01-sz1.jpg --top_k 5
```

**What to highlight in the results:**
> "Here you can see the deep learning branch in action. We queried this sketch, and in about 3 seconds, the system retrieved the top 5 closest matches from our gallery. Notice that the Rank 1 match has the lowest L2 distance. If we look at the corresponding photo, we can see the system successfully bridged the gap between the sketch and the real person."

**Action:**
- Briefly open the Rank 1 photo: `data/cufs/photos/f-041-01.jpg` (or whichever photo is returned as Rank 1) to prove the match visually.

---

## 🧠 Part 2: The Full Architecture & Fusion (2 mins)

**What to say:**
> "While deep learning is powerful, sketches can have severe distortions. To make the system robust, we use a fused pipeline. We combine our deep embeddings with a hand-crafted feature branch (HOG + Spearman distance). Let's query the same sketch through the full pipeline."

**Action:**
- Open `pipeline.py` and briefly point to the `match()` function, specifically where the deep and hand-crafted scores are fused (look for the min-max normalization).

**Command to run:**
```bash
python3 pipeline.py match --sketch data/cufs/sketches/m-063-01-sz1.jpg --mode viewed
```

**What to highlight in the results:**
> "Notice the output now gives us a `fused_distance`. This single score is the normalized combination of both branches, exactly mirroring the paper's dual-branch architecture. This fusion is critical for handling real-world forensic sketches."

---

## 📊 Part 3: The Headline Metrics (1.5 mins)

**What to say:**
> "A system is only as good as its benchmark. Let's evaluate our deep embedding model across the entire held-out test set of 30 identities."

**Command to run:**
```bash
python3 evaluate.py
```

**What to highlight in the results:**
> "Look at the Rank-1 accuracy: **90%**. This means that 90% of the time, the absolute first photo returned by the system is the correct suspect. By Rank-5, we hit 100%. This proves the InceptionResnetV1 backbone, fine-tuned with our triplet loss, is highly effective at mapping sketches and photos into the same latent space."

---

## 🌟 Part 4: The "Wow" Factor - 3D Morphable Models (3 mins)

**What to say:**
> "Now for the most advanced part of the pipeline. To handle extreme variations in pose and expression, we implemented synthetic data generation using the state-of-the-art **Basel Face Model 2019 (BFM)**."

**Action:**
- Open `render_3dmm.py`. Briefly scroll through the code to show it's a pure-Python implementation loading the `model2019_bfm.h5` file and projecting 3D vertices to 2D.

**What to say:**
> "Instead of relying on clunky Java or MATLAB pipelines, we built a pure Python renderer that directly samples the PCA spaces for shape, color, and expression to generate 3D face variants."

**Command to run:**
```bash
python3 render_3dmm.py --k 9 --output_dir 3dmm_renders --manifest my_3dmm_renders.csv
```

**What to highlight while it runs (takes ~40s):**
> "Right now, the system is dynamically generating 270 synthetic face images—9 pose and expression variants for each of our 30 test identities."

**Action:**
- Once it finishes, open the file `3dmm_sample_grid.png`.

**What to highlight in the results:**
> *(With the image open)* "This is the result. From the mathematical model, we generated these 3D faces with varying yaw rotations (e.g., -20 to +20 degrees) and expressions. By importing these into our multi-sketch fusion pipeline, we can simulate different camera angles and expressions that might match a forensic sketch, significantly enriching our dataset and retrieval capabilities."

---

## 🏁 Conclusion (30 seconds)

**What to say:**
> "In summary, we've built a complete, end-to-end forensic retrieval system. We successfully implemented the deep learning backbone, the hand-crafted feature fusion, and integrated advanced 3D Morphable Model synthesis using BFM 2019. 
>
> Thank you for watching the demo."
