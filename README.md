# MoBYv2AL
The first contrastive learning work for Active Learning.
Link to paper: https://arxiv.org/abs/2301.01531
Please install the required libraries from requirements.txt
Pip install requirements:
```
pip install -r requirements.txt
```
To run MoBYv2AL (also Random and CoreSet) on CIFAR-10 (also CIFAR-100/SVHN/FashionMNIST): 
```
python main_ssl.py -d cifar10 -m mobyv2al
```
If you find this code useful please cite:

## Citation

```bibtex
@booktitle{caramalau2022mobyv2al,
  title={MoBYv2AL: Self-supervised Active Learning for Image Classification},
  author={Caramalau, Razvan and Bhatarrai, Binod and Stoyanov, Dan and Kim, Tae-Kyun},
  booktitle={BMVC (British Machine Vision Conference)},
  year={2022}
}
```
