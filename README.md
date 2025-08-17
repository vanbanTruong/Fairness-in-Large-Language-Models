# Fairness in Language Models

This ongoing project aims to consolidate interesting efforts in the field of fairness in Language Models (LMs), drawing on the proposed taxonomy and surveys dedicated to various aspects of fairness in LMs.

**Disclaimer: We may have missed some relevant papers in the list. If you have suggestions or want to add papers, please submit a pull request or email us—your contributions are greatly appreciated!**

**Tutorial:** [Fairness in Language Models: A Tutorial](https://fairness-llms-tutorial.github.io)<br>
Zichong Wang, Avash Palikhe, Zhipeng Yin, Jiale Zhang and Wenbin Zhang<br>
*The 34<sup>th</sup> International Joint Conference on Artificial Intelligence (IJCAI), Montreal, Canada, 2025*

**Introduction to LMs:** [History, Development, and Principles of Large Language Models-An Introductory Survey
](https://arxiv.org/abs/2402.06853)<br>
Zichong Wang, Zhibo Chu, Thang Viet Doan, Shiwen Ni, Min Yang and Wenbin Zhang<br>
*AI and Ethics, 2025*

**Fairness in LMs:** [Fairness in Large Language Models: A Taxonomic Survey](https://dl.acm.org/doi/abs/10.1145/3682112.3682117)<br>
Zhibo Chu, Zichong Wang and Wenbin Zhang<br>
*ACM SIGKDD Explorations Newsletter, 2024*

**Fairness Definitions in LMs:** [Fairness Definitions in Language Models Explained](https://arxiv.org/abs/2407.18454)<br>
Avash Palikhe, Zichong Wang, Zhipeng Yin and Wenbin Zhang

**Datasets for Fairness in LMs:** [Datasets for Fairness in Language Models: An In-Depth Survey](https://arxiv.org/abs/2506.23411)<br>
Jiale Zhang, Zichong Wang, Avash Palikhe, Zhipeng Yin and Wenbin Zhang


Email: ziwang@fiu.edu - Zichong Wang<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wenbinzhang2008@gmail.com - Wenbin Zhang<br>

---
### 📚 **Contents**

- [Mitigating Bias in LMs](#mitigating-bias-in-lms-link-to-the-paper)
  - [By Year](#by-year)
  - [By Category](#by-category)
- [Quantifying Bias in LMs](#quantifying-bias-in-lms-link-to-the-paper)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Introduction to LMs ([Link](https://link.springer.com/article/10.1007/s43681-024-00583-7) to the paper)
![Fairness in Large Language Models](https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/blob/main/tutorial/images/Introduction_to_LMs.png)


## Fairness in LMs ([Link](https://dl.acm.org/doi/abs/10.1145/3682112.3682117) to the paper)
![Fairness in Large Language Models](https://github.com/super-hash/Fairness-in-Large-Language-Models/blob/main/tutorial/images/Fairness%20in%20Large%20Language%20Models.png)

## Quantifying Bias in LMs ([Link](https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/definitions) to the repository)
> <div align="justify">
> 
> This repository systematizes fairness definitions and bias quantification methods for language models across different transformer architectures. It provides a comprehensive taxonomy that categorizes fairness notions based on encoder-only, decoder-only, and encoder-decoder model types, together with clear mathematical formulations, empirical demonstrations, and practical implementation guidelines to support consistent and architecture-appropriate fairness evaluations in language model research.
> 
> </div>
<!-- ![Fairness Definitions in LMs](https://github.com/user-attachments/assets/884f134c-ee31-4c03-9487-2907738b77f8) -->
![Fairness Definitions in LMs](tutorial/images/Fairness%20definitions%20for%20LMs.jpg)


## Datasets ([Link](https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/datasets) to the repository)
> <div align="justify">
> 
> This repository aggregates and systematizes benchmark datasets used to evaluate fairness and social bias in language models (LMs). It provides a unified taxonomy and rich metadata describing each dataset’s structure, provenance, language coverage, bias types, and accessibility, together with reproducible code and standardized evaluation pipelines to support transparent, comparable fairness audits across models and tasks.
> 
> </div>
![Screenshot 2024-10-07 at 1 52 35 PM](https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/blob/main/tutorial/images/datasets_taxonomy.png)



## Citation
### Fairness in Large Language Models: A Taxonomic Survey [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://dl.acm.org/doi/abs/10.1145/3682112.3682117)

If you find that our taxonomic survey helps your research, we would appreciate citations to the following paper:
```
@article{chu2024fairness,
  title={Fairness in Large Language Models: A Taxonomic Survey},
  author={Chu, Zhibo and Wang, Zichong and Zhang, Wenbin},
  journal={ACM SIGKDD Explorations Newsletter},
  volume={26},
  number={1},
  pages={34--48},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

### History, Development, and Principles of Large Language Models-An Introductory Survey [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/pdf/2402.06853)

If you find that our introduction survey helps your research, we would appreciate citations to the following paper:
```
@article{wang2024history,
  title={History, Development, and Principles of Large Language Models: An Introductory Survey},
  author={Wang, Zichong and Chu, Zhibo and Doan, Thang Viet and Ni, Shiwen and Yang, Min and Zhang, Wenbin},
  journal={AI and Ethics},
  year={2024},
  publisher={Springer}
}

```

### Fairness Definitions in Language Models Explained [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/abs/2407.18454)

If you find that our definition survey helps your research, we would appreciate citations to the following paper:
```
@article{palikhe2024fairness,
  title={Fairness definitions in language models explained},
  author={Palikhe, Avash and Wang, Zichong and Yin, Zhipeng and Zhang, Wenbin},
  journal={arXiv preprint arXiv:2407.18454},
  year={2024}
}

```

### Datasets for Fairness in Language Models: An In-Depth Survey [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/abs/2506.23411)

If you find that our dataset survey helps your research, we would appreciate citations to the following paper:
```
@misc{zhang2025datasetsfairnesslanguagemodels,
      title={Datasets for Fairness in Language Models: An In-Depth Survey}, 
      author={Jiale Zhang and Zichong Wang and Avash Palikhe and Zhipeng Yin and Wenbin Zhang},
      year={2025},
      eprint={2506.23411},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.23411}, 
}


