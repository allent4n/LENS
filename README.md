
<div align="center">

# [IPM] Localize-then-summarize: Enhancing scientific multimodal summarization with facet-aware cross-modal memory


[![Paper](https://img.shields.io/badge/paper-IPM-blue)](https://www.sciencedirect.com/science/article/pii/S030645732600378X?via%3Dihub)
![Python](https://img.shields.io/badge/python-3.8-blue)

</div>

Official implementation of [Localize-then-summarize: Enhancing scientific multimodal summarization with facet-aware cross-modal memory](https://www.sciencedirect.com/science/article/pii/S030645732600378X?via%3Dihub)


## ❓ What is LENS

Scientific papers follow structured facets (e.g., Introduction, Methods), and modern research dissemination increasingly incorporates multimodal formats like presentation videos and audio. This shift necessitates summarization systems that can process both structured and multimodal information. This work proposes Localize-then-Summarize (LENS), a two-stage scientific summarizer that first localizes relevant presentation segments that align with paper facets; followed by summarizing them via memory-augmented reasoning that models dependencies across modalities and facets. On a new MFS-SciSum dataset with 2.7k aligned paper–presentation pairs, the LENS localizer and summarizer achieve Recall@0.5/0.7 scores of 40.83/23.06, and ROUGE-1/2/L scores of 44.71/15.26/21.64, outperforming strong baselines like CLIP and Transformer by 10–15 points in Recall and 1–5 points in ROUGE (resp.). Additionally, the LENS summarizer reduces GPU usage by 71%, notably improving generation efficiency.
<img width="696" height="518" alt="image" src="https://github.com/user-attachments/assets/bdb7fa57-0bf9-4464-bd8f-701fd3a09869" />


## ⚡️ Quickstart
1. **Clone the GitHub Repository:** 

   ```shell
   git clone https://github.com/allent4n/LENS
   ```

2. **Set Up Python Environment:** 

   ```shell
   cd LENS
   conda env create -f environment.yml
   conda activate LENS
   ```

3. **Install SEA Dependencies:** 
   ```shell
   conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install git+https://github.com/openai/CLIP.git
   pip install openai-whisper sentence-transformers==2.2.2 tokenizers==0.20.3 transformers==4.46.3 accelerate==1.0.1 peft==0.13.2
   pip install git+https://github.com/bckim92/language-evaluation.git
   pip install gdown pickle5 rouge==1.0.1 rouge_score==0.1.2 srt kornia boto3 pandas pycocoevalcap timm
   ```
   
4. **Reproduce Results:**
   
   ```shell
   bash reproduce.sh
   ```

## 🔎 Citation

```
@article{tan2027localize,
  title={Localize-then-summarize: Enhancing scientific multimodal summarization with facet-aware cross-modal memory},
  author={Tan, Zusheng and Ji, Jing-Yu and Yu, Wenhui and Ng, Ngai Fung and Yang, Fan and Tang, Jeff and Fong, Ken and Li, Jing and Kwong, Sam and Chiu, Billy},
  journal={Information Processing \& Management},
  volume={64},
  number={1},
  pages={104987},
  year={2027},
  publisher={Elsevier}
}
```


## 📬 Contact

If you have any inquiries, suggestions, or wish to contact us for any reason, we warmly invite you to email us at allentan@ln.hk.
