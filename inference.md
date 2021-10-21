## PororoMRC Inference
- Baseline과 거의 비슷하고, run_mrc/main 함수만 조금 수정했습니다.
- Scoring을 위해 변형하기 위해 PororoMRC관련 Class 추가

```python
import logging
import sys
import json
from typing import Callable, List, Dict, NoReturn, Tuple

import numpy as np
from tqdm import tqdm

from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from pororo import Pororo

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from typing import Dict, Tuple, Union

import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer

from pororo.models.brainbert import BrainRobertaModel

logger = logging.getLogger(__name__)


class BrainRobertaModel_sub(BrainRobertaModel):
    """
    Helper class to load pre-trained models easily. And when you call load_hub_model,
    you can use brainbert models as same as RobertaHubInterface of fairseq.
    Methods
    -------
    load_model(log_name: str): Load RobertaModel

    """

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        """
        Load pre-trained model as RobertaHubInterface.
        :param model_name: model name from available_models
        :return: pre-trained model
        """
        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        tok_path = download_or_load(f"tokenizers/bpe32k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BrainRobertaHubInterface_sub(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )


class BrainRobertaHubInterface_sub(RobertaHubInterface):
    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json",
            merges_filename=f"{tok_path}/merges.txt",
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (
                " " + self.tokenize(s, add_special_tokens=False) + " </s>"
                if add_special_tokens
                else ""
            )
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def decode(
        self,
        tokens: torch.LongTensor,
        skip_special_tokens: bool = True,
        remove_bpe: bool = True,
    ) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos() and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [
                np.array([c for c in s if c != self.task.source_dictionary.eos()])
                for s in sentences
            ]

        sentences = [
            " ".join([self.task.source_dictionary.symbols[c] for c in s])
            for s in sentences
        ]

        if remove_bpe:
            sentences = [
                s.replace(" ", "").replace("▁", " ").strip() for s in sentences
            ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_output(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        show_probs: bool = False,
    ) -> Union[str, Dict]:
        """Predict output, either a classification label or regression target,
         using a fine-tuned sentence prediction model.
        :returns output
            str (classification) or float (regression)
            >>> from brain_bert import BrainRobertaModel
            >>> model = BrainRobertaModel.load_model('brainbert.base.ko.kornli')
            >>> model.predict_output(
            ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
            ...    'BrainBert는 한국어 모델이다.',
            ...    )
            entailment
            >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korsts')
            >>> model.predict_output(
            ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
            ...    'BrainBert는 한국어 모델이다.',
            ...    )
            0.8374465107917786
        """
        assert self.args.task == "sentence_prediction", (
            "predict_output() only works for sentence prediction tasks.\n"
            "Use predict() to obtain model outputs; "
            "use predict_span() for span prediction tasks."
        )
        assert (
            "sentence_classification_head" in self.model.classification_heads
        ), "need pre-trained sentence_classification_head to make predictions"

        tokens = self.encode(
            sentence,
            *addl_sentences,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        with torch.no_grad():
            prediction = self.predict(
                "sentence_classification_head",
                tokens,
                return_logits=self.args.regression_target,
            )
            if self.args.regression_target:
                return prediction.item()  # float

            label_fn = lambda label: self.task.label_dictionary.string(
                [label + self.task.label_dictionary.nspecial]
            )

            if show_probs:
                probs = softmax(prediction.cpu().numpy())
                probs = probs.tolist()
                probs = {label_fn(i): prob for i, prob in enumerate(probs)}
                return probs

        return label_fn(prediction.argmax().item())  # str

    @torch.no_grad()
    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:
        """
        Predict span from context using a fine-tuned span prediction model.

        :returns answer
            str

        >>> from brain_bert import BrainRobertaModel
        >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korquad')
        >>> model.predict_span(
        ...    'BrainBert는 어떤 언어를 배운 모델인가?',
        ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
        ...    )
        한국어

        """

        max_length = self.task.max_positions()
        tokens = self.encode(
            question,
            context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )[:max_length]
        with torch.no_grad():
            logits = self.predict(
                "span_prediction_head",
                tokens,
                return_logits=True,
            ).squeeze()  # T x 2
            # first predict start position,
            # then predict end position among the remaining logits
            start = logits[:, 0].argmax().item()
            mask = (
                torch.arange(logits.size(0), dtype=torch.long, device=self.device)
                >= start
            )
            end = (mask * logits[:, 1]).argmax().item()
            # end position is shifted during training, so we add 1 back
            answer_tokens = tokens[start : end + 1]

            answer = ""
            if len(answer_tokens) >= 1:
                decoded = self.decode(answer_tokens)
                if isinstance(decoded, str):
                    answer = decoded

        return (
            logits[:, 0][logits[:, 0].argmax()],
            logits[:, 1][(mask * logits[:, 1]).argmax()],
        ), (answer, (start, end + 1))

    @torch.no_grad()
    def predict_tags(
        self,
        sentence: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ):
        tokens = self.encode(
            sentence,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        label_fn = lambda label: self.task.label_dictionary.string([label])

        # Get first batch and ignore <s> & </s> tokens
        preds = (
            self.predict(
                "sequence_tagging_head",
                tokens,
            )[0, 1:-1, :]
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
        labels = [
            label_fn(int(pred) + self.task.label_dictionary.nspecial) for pred in preds
        ]
        return [
            (
                token,
                label,
            )
            for token, label in zip(self.tokenize(sentence).split(), labels)
        ]


from typing import Optional, Tuple
from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase


class PororoMrcFactory(PororoFactoryBase):
    """
    Conduct machine reading comprehesion with query and its corresponding context

    Korean (`brainbert.base.ko.korquad`)

        - dataset: KorQuAD 1.0 (Lim et al. 2019)
        - metric: EM (84.33), F1 (93.31)

    Args:
        query: (str) query string used as query
        context: (str) context string used as context

    Returns:
        Tuple[str, Tuple[int, int]]: predicted answer span and its indices

    Examples:
        >>> mrc = Pororo(task="mrc", lang="ko")
        >>> mrc(
        >>>    "카카오브레인이 공개한 것은?",
        >>>    "카카오 인공지능(AI) 연구개발 자회사 카카오브레인이 AI 솔루션을 첫 상품화했다. 카카오는 카카오브레인 '포즈(pose·자세분석) API'를 유료 공개한다고 24일 밝혔다. 카카오브레인이 AI 기술을 유료 API를 공개하는 것은 처음이다. 공개하자마자 외부 문의가 쇄도한다. 포즈는 AI 비전(VISION, 영상·화면분석) 분야 중 하나다. 카카오브레인 포즈 API는 이미지나 영상을 분석해 사람 자세를 추출하는 기능을 제공한다."
        >>> )
        ('포즈(pose·자세분석) API', (33, 44))
        >>> # when mecab doesn't work well for postprocess, you can set `postprocess` option as `False`
        >>> mrc("카카오브레인이 공개한 라이브러리 이름은?", "카카오브레인은 자연어 처리와 음성 관련 태스크를 쉽게 수행할 수 있도록 도와 주는 라이브러리 pororo를 공개하였습니다.", postprocess=False)
        ('pororo', (30, 34))

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.korquad"]}

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "brainbert" in self.config.n_model:
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            # from pororo.models.brainbert import BrainRobertaModel
            from pororo.utils import postprocess_span

            model = (
                BrainRobertaModel_sub.load_model(
                    f"bert/{self.config.n_model}",
                    self.config.lang,
                )
                .eval()
                .to(device)
            )

            tagger = mecab.MeCab()

            return PororoBertMrc(model, tagger, postprocess_span, self.config)


class PororoBertMrc(PororoBiencoderBase):
    def __init__(self, model, tagger, callback, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._callback = callback

    def predict(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Conduct machine reading comprehesion with query and its corresponding context

        Args:
            query: (str) query string used as query
            context: (str) context string used as context
            postprocess: (bool) whether to apply mecab based postprocess

        Returns:
            Tuple[str, Tuple[int, int]]: predicted answer span and its indices

        """
        postprocess = kwargs.get("postprocess", True)

        score, pair_result = self._model.predict_span(query, context)
        span = (
            self._callback(
                self._tagger,
                pair_result[0],
            )
            if postprocess
            else pair_result[0]
        )

        return (
            score,
            span,
            pair_result[1],
        )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Pororo를 이용할 경우, Config, Model 불필요
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:  # trainset에 대한 evaluation을 진행
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )

    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, datasets, tokenizer)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> NoReturn:

    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    eval_dataset = datasets["validation"]

    predictions = {}
    mrc = PororoMrcFactory(task="mrc", lang="ko", model=None)
    mrc_predict = mrc.load(device="cpu")

    for i in tqdm(range(len(eval_dataset))):
        _id = eval_dataset["id"][i]
        context = eval_dataset["context"][i]
        question = eval_dataset["question"][i]
        one_predictions = []
        for cont in context.split("♧"):
            score, span, pair_result = mrc_predict(question, cont, postprocess=False)
            one_predictions.append({"score": score[0] + score[1], "span": span})
        one_predictions = sorted(
            one_predictions, key=lambda x: x["score"], reverse=True
        )[0]

        predictions[_id] = one_predictions["span"]

    def post_processing_function(predictions):
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    if training_args.do_predict:
        # predictions = post_processing_function(predictions)
        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

        with open("./outputs/_tmp/prediction.json", "w", encoding="utf-8") as writer:
            writer.write(json.dumps(predictions, indent=4, ensure_ascii=False) + "\n")

    if training_args.do_eval:
        predictions = post_processing_function(predictions)
        metrics = compute_metrics(predictions)
        metrics["eval_samples"] = len(eval_dataset)
        print("metrics: ", metrics)


if __name__ == "__main__":
    main()
```
