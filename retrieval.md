

## pororo

- huggingface hub을 0.0.19로 버전을 낮춰야 한다.
- mecab 활용할 때 GCC 연동하는 문제가 생겼었다. 

위의 문제들을 해결하려면 다음과 같이 입력하면 된다.

```bash
pip install huggingface-hub==0.0.19
pip install pororo

# pororo mrc
apt-get install build-essential -y
pip install python-mecab-ko
```

- `from konlpy.tag import ``Mecab` 부분에서
  `AttributeError: module 'tweepy' has no attribute 'StreamListener'` 와 같은 에러를 만났는데요.
  tweepy를 3.9.0버전으로 설치하면 해결됩니다!
  `pip install tweepy==3.9.0`

