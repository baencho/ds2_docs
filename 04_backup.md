# 📚 04. Pruning and Sparsity for LLMs

Lecture 2, 3이 일반적인 neural network pruning을 다뤘다면, Lecture 4는 그걸 **LLM에 적용할 때 무엇이 달라지는지**, 그리고 **LLM inference를 효율적으로 만들기 위해 sparsity를 시스템이 어떻게 지원하는지**를 다뤄. 강의 목표도 “efficient LLM inference를 위해 system이 sparsity를 어떻게 지원하는지 이해하는 것”이고, 내용은 **Weight Sparsity, Activation Sparsity, Attention Sparsity**로 나뉘어 있어. 

---

## 4-1. Lecture 4 전체 구조

Lecture 4의 큰 주제는 세 가지야.

| 구분                  | 핵심 내용                        |
| ------------------- | ---------------------------- |
| Weight Sparsity     | Wanda, M:N Sparsity, EIE     |
| Activation Sparsity | MoE, DeepSeekMoE             |
| Attention Sparsity  | SpAtten, Quest, DuoAttention |

즉 단순히 weight만 자르는 게 아니라, LLM에서 생기는 sparsity를 여러 위치에서 활용하려는 강의야.

---

## 4-2. 일반 magnitude pruning의 한계

기존 pruning **magnitude pruning**: weight의 절댓값이 작은 것을 제거하는 방식이야.

$$
|w_1| < |w_2| \Rightarrow w_1 \text{을 pruning}
$$

![0](image.png)

그런데 LLM에서는 이게 잘 안 맞을 수 있다

왜냐하면 실제 linear layer의 영향은 weight만이 아니라 **input activation**과 같이 결정되기 때문이다.

특히 activation에 **outlier channel**, 즉 특정 channel 값이 유난히 크게 튀는 현상이 생길 수 있다.

(activation 값이 엄청 커서 다른 것보다 weight 은 작은데 둘을 곱했을 땐 다른 것보다 커지는 경우)

---

## 4-3. Wanda: Weights and Activations-Aware Pruning

이 문제를 해결하려는 방법이 **Wanda**야.

![1](image-1.png)

Wanda의 핵심 아이디어는:

> weight 크기만 보지 말고, activation 크기도 같이 보자.

그래서 pruning importance score를 이렇게 잡아.

$$
S = |W| \cdot |X|
$$

즉 각 weight의 중요도를 단순히 $|W|$로 보는 게 아니라,

$$
\text{importance} = |\text{weight}| \times |\text{activation}|
$$

으로 보는 거야.

쉽게 말하면:

> weight가 작더라도 그 weight가 곱해지는 activation이 크면 중요할 수 있다.

---

## 4-4. M:N Sparsity

다음은 **M:N Sparsity**야.

![2](image-2.png)

이건 완전히 자유롭게 weight를 자르는 fine-grained pruning과 다르게, **정해진 block 안에서 일정 개수만 남기는 structured sparsity**야.

대표적인 예가 NVIDIA의 **2:4 sparsity**야.

의미는:

> weight 4개 중 2개만 non-zero로 남기고, 나머지 2개는 0으로 만든다.

일반화하면 M:N sparsity는:

> N개 중 M개만 남긴다.

예를 들어:

$$
2:4 \text{ sparsity}
$$

는 4개 중 2개만 남기는 거야.

---

## 4-1. 왜 M:N sparsity를 쓰는가?

Fine-grained pruning은 weight를 아무 위치나 자를 수 있어.

장점은 성능 유지가 좋을 수 있다는 것.
하지만 단점은 weight가 불규칙하게 남아서 hardware가 빠르게 계산하기 어렵다는 것.

반면 M:N sparsity는 규칙이 있어.

예를 들어 매 4개 weight마다 2개만 남긴다고 정해져 있으면, hardware가 이 패턴을 알고 빠르게 계산할 수 있어.

Lecture 4에서도 M:N sparsity는 nonzero element를 메모리의 왼쪽으로 모으고, index metadata를 저장해서 storage와 computation을 줄이는 구조로 설명돼. 

---

## 4-2. M:N sparsity에는 hardware 지원이 필요함

M:N sparsity가 실제로 빨라지려면 hardware가 이 sparse pattern을 이해해야 해.

예를 들어 2:4 sparsity에서는 4개 중 2개만 곱셈하면 돼.

$$
4 \text{ multiplications} \rightarrow 2 \text{ multiplications}
$$

하지만 문제는 남은 2개가 원래 input의 어느 위치와 곱해져야 하는지 알아야 한다는 거야.

그래서 index 또는 metadata가 필요해.

Lecture 4에서는 NVIDIA Tensor Core가 M:N sparse matrix를 처리할 수 있도록, nonzero value와 index metadata를 함께 사용한다고 설명해. 즉 input의 어느 position을 mask할지 hardware logic이 필요하다는 거야. 

---

## 4-5. EIE: Efficient Inference Engine (덜 중요)

EIE는 **Efficient Inference Engine**의 약자야.

fine-grained pruning은 임의의 weight가 제거되기 때문에 acceleration이 어렵고, memory access도 불규칙해질 수 있다

-> EIE는 "fine-grained pruning으로 생긴 sparse weight를 효율적으로 처리하려는 hardware accelerator"이다

현재 LLM 관점에서는 M:N sparsity처럼 더 hardware-friendly한 구조가 더 강조되는 느낌이야.

> 요지는 fine-grained pruning 을 해도 특별한 HW 구조를 만들면 acceleration 이 된다

---

## 4-6. Activation Sparsity: MoE

### 4-6-1. 배경
![3](image-3.png)
- FFN 은 weight 가 많아서 scale up 하기 어렵다
- input token 에 맞는 FFN parallelization 을 하자 = MoE

### 4-6-2. MoE
이제 weight sparsity가 아니라 **activation sparsity**로 넘어가.

대표적인 방법이 **MoE**, 즉 **Mixture of Experts**야.

![4](image-4.png)

MoE는 모델 안에 여러 expert network를 두고, 입력 token마다 일부 expert만 선택해서 사용해.

예를 들어 expert가 8개 있다고 해보자.

$$
E_1, E_2, \dots, E_8
$$

일반 dense model이면 모든 parameter를 매번 사용하지만, MoE는 token마다 top-$k$ expert만 사용해.

예를 들어 top-$2$ routing이면:

$$
\text{token} \rightarrow E_3, E_7
$$

이렇게 일부 expert만 활성화돼.

### 4-6-3. Routing Capacity

![5](image-5.png)

**MoE(Mixture of Experts)**에서 token들을 expert들에게 보낼 때, **각 expert가 한 batch에서 최대 몇 개 token까지 처리할 수 있는지**를 설명하는 거야.

> MoE에서는 모든 token이 모든 expert를 거치는 게 아니라, router가 token마다 적절한 expert를 골라 보낸다.
> 그런데 token이 특정 expert에게 너무 몰리면 병렬성이 깨지고 load imbalance가 생긴다.
> 그래서 expert마다 처리 가능한 최대 token 수, 즉 **capacity**를 정해둔다.

- token이 총 6개 있고, expert가 3개 있어.
- 평균적으로는 expert 하나당 token이 2개 가는 게 이상적이야.

**Capacity factor**

$$
C =
\left(
\frac{\text{tokens per batch}}{\text{number of experts}}
\right)
\times
\text{capacity factor}
$$

여기서 $C$는 **expert 하나가 처리할 수 있는 최대 token 수**야.

즉 capacity factor는 expert마다 평균보다 얼마나 여유 공간을 더 줄지 정하는 값이야.

- Capacity factor = 1일 때
  - token 6개, expert 3개, capacity factor가 1이면:
  - $C = \frac{6}{3} \times 1 = 2$
  - expert 하나는 최대 2개 token만 처리할 수 있어.
  - 그런데 router가 token들을 보냈는데, 만약 Expert 1에게 token이 3개 몰렸다고 해보자.
  - Expert 1의 capacity는 2니까, 3개 중 2개만 처리하고 나머지 1개는 처리할 수 없어.
  - C=1, an expert processes at most $6/3 \times 1 = 2$ tokens; one token is skipped
  - 즉 **expert가 꽉 차서 overflow된 token 하나가 skip/drop된다**는 뜻이야.

- Capacity factor = 1.5일 때
  - $C = \frac{6}{3} \times 1.5 = 3$
  - 즉 expert 하나가 최대 3개 token까지 처리할 수 있어.
  - 그러면 Expert 1에게 token이 3개 몰려도 처리 가능해.
  - if C=1.5, an expert processes at most $6/3 \times 1.5 = 3$ tokens; slack for expert 2&3
  - 즉 Expert 1은 3개를 처리할 수 있고, Expert 2와 Expert 3은 capacity가 3인데 실제로 2개, 1개만 받을 수도 있으니까 빈 공간이 생겨.
  - 이 빈 공간을 **slack**이라고 해.

capacity factor와 성능:

- capacity factor가 너무 작으면 좋은 expert가 있어도 token을 못 보냄
- 예를 들어 어떤 token에게는 Expert 1이 가장 적합한데, Expert 1이 이미 꽉 찼으면 그 token은 skip되거나 덜 적합한 expert로 가야 해.
- 그러면 모델 성능이 떨어질 수 있어.
- capacity factor를 키우면 expert마다 여유 공간이 생겨서, token이 원래 가야 할 expert로 갈 가능성이 커져.
- 그래서 성능은 좋아질 수 있어.
- 반대로 capacity factor를 너무 크게 하면 단점도 있어.
- 각 expert가 더 많은 token을 받을 수 있도록 buffer를 크게 잡아야 하니까 메모리와 계산 여유분이 늘어나.
- 또 expert가 여러 device에 나뉘어 있으면, token을 expert가 있는 device로 보내야 하고 token 을 다른 device 로 옮기는 communication cost (**Across Device Communication**) 발생

### 4-6-4. Routing Policy
![6](image-6.png)

### 4-6-5. MoE가 sparse한 이유

MoE는 parameter 전체는 엄청 클 수 있어.

하지만 한 token이 지나갈 때는 모든 expert를 쓰지 않고 일부 expert만 써.

즉 전체 parameter 수는 크지만, 실제 token당 계산량은 제한돼.

이걸 sparse activation이라고 볼 수 있어.

> 많은 expert 중 일부만 activate되기 때문.

그래서 MoE의 장점은:

> 모델 capacity는 키우면서, token당 computation은 덜 늘릴 수 있다.

LLM에서 큰 모델을 만들 때 매우 중요한 아이디어야.

### 4-6-6. MoE의 어려움

MoE는 좋아 보이지만 시스템적으로 어렵다.

왜냐하면 token마다 선택되는 expert가 다를 수 있기 때문이야.

예를 들어 batch 안의 token들이 각각 다른 expert로 가면, 계산이 불규칙해지고 load balancing 문제가 생겨.

어떤 expert는 token이 몰려서 바쁘고, 어떤 expert는 거의 안 쓰일 수 있어.

그래서 MoE 시스템에서는 routing, load balancing, expert parallelism 같은 문제가 중요해.

---

## 4-7. DeepSeekMoE
Lecture 4에서는 activation sparsity의 예시로 **DeepSeekMoE**도 들어가.

### 4-7-1. A few large experts 구조의 문제

![7](image-7.png)

- Knowledge Hybrity: expert 가 상관없는 지식도 갖고 있음 
- Knowledge Redundancy: 특화된 지식이 아니라 일반적 지식을 중복해서 갖고 있다. 

### 4-7-2. DeepSeekMoE는 두 가지 방법 사용

![8](image-8.png)

1. Fine Grained Expert Segmentation: 작은 expert 들은 둔다
2. Shared Expert Isolation: 몇 개의 common knowledge 를 가진 shared expert 도 둔다
3. 그리고 이런 여러 expert 들을 조합해서 쓴다

---

## 4-8. Attention Sparsity

Transformer에서 attention은 매우 비싼 연산이야.

Self-attention의 기본 복잡도는 sequence length를 $n$이라고 하면:

$$
O(n^2)
$$

왜냐하면 모든 token이 모든 token을 보기 때문이야.

sequence가 길어질수록 attention 계산량과 memory가 급격히 증가해.

그래서 attention에서도 sparsity를 활용하려는 방법들이 나온다.

예시는:

* SpAtten
* Quest
* DuoAttention

이야. 

---
<div style="page-break-inside: avoid; break-inside: avoid;">

## 4-9. SpAtten

SpAtten은 attention에서 중요하지 않은 부분을 줄이는 방식이야.

![9](image-9.png)

기본 아이디어는:

> 모든 token pair를 다 보지 말고, 중요한 attention만 남기자.

즉 attention score나 head/token importance를 보고 덜 중요한 부분을 pruning해서 attention computation을 줄이려는 방법이야.

Transformer attention은 많은 token 간 관계를 계산하지만, 실제로는 모든 관계가 똑같이 중요한 건 아니야.

어떤 token은 거의 참고되지 않을 수도 있고, 어떤 attention head는 덜 중요할 수도 있어.

SpAtten은 이런 sparsity를 이용해 attention computation과 memory cost를 줄이려는 방법으로 이해하면 돼.

![10](image-10.png)

근데 문제는 그냥 이렇게 고정적으로 특정 attention 을 버리면 나중에 혹시 필요하게 됐을 때 (다른 Q가 왔을 때) 참고할 수 없음

</div>
---

<div style="page-break-inside: avoid; break-inside: avoid;">

## 4-10. Quest

![11](image-11.png)

Quest도 long-context LLM에서 attention cost를 줄이려는 방법으로 보면 돼.

긴 context에서는 모든 token을 다 보는 것이 너무 비싸기 때문에, query와 관련 있는 key/value만 골라서 attention을 수행하려는 방향이야.

즉 기본 아이디어는:

> 현재 query에 중요한 token만 찾아서 attention하자.

이렇게 하면 attention이 dense한 $n \times n$ 계산이 아니라, 필요한 일부 token에 대해서만 수행될 수 있어.

---

## 4-11. DuoAttention

![12](image-12.png)

DuoAttention은 attention의 역할을 나누어 효율화하는 방법으로 볼 수 있어.

LLM attention에서는 모든 head가 같은 방식으로 long context 전체를 봐야 하는 것은 아닐 수 있어.

어떤 head는 global context를 봐야 하고, 어떤 head는 local context만 봐도 충분할 수 있어.

DuoAttention은 이런 차이를 이용해서 attention computation과 KV cache 사용량을 줄이려는 방법으로 이해하면 돼.

![13](image-13.png)

장점:
- KV memory 와 decoding latency 가 줄어든다
- long-context accuracy 유지

</div>

## 4-15. 시험/과제용으로 외울 핵심

Lecture 4에서 꼭 기억해야 할 건 이거야.

1. **LLM에서는 magnitude pruning만으로 부족하다.**
   weight가 작아도 activation이 크면 실제 영향이 클 수 있다.

2. **Wanda는 weight와 activation을 같이 본다.**

   $$
   S = |W| \times |X|
   $$

3. **M:N sparsity는 hardware-friendly structured sparsity다.**
   예: $2:4$ sparsity는 4개 중 2개만 남긴다.

4. **Sparse하다고 무조건 빨라지는 건 아니다.**
   system / hardware support가 있어야 실제 speedup이 나온다.

5. **LLM sparsity는 weight만이 아니라 activation, attention에도 있다.**

6. **MoE는 activation sparsity의 대표 예시다.**
   많은 expert 중 일부만 활성화한다.

7. **Attention sparsity는 long context에서 중요하다.**
   attention은 기본적으로 $O(n^2)$라서 sequence가 길어질수록 부담이 커진다.

---

<div style="page-break-after: always;"></div>

# 📚 05. Quantization (Part I)

quantization은 **숫자를 더 적은 bit로 표현해서 memory와 연산 비용을 줄이는 방법**이야. Lecture 5의 목표도 neural network quantization의 기본 개념과 대표적인 quantization 방법을 이해하는 것이고, 내용은 **data type 복습 → quantization 기본 → K-means quantization → linear quantization → binary/ternary quantization** 순서로 구성돼 있어. 

---

## 5-1. Quantization이 왜 필요한가?

딥러닝 모델은 weight와 activation을 보통 floating-point로 저장하고 계산해.

예를 들어 일반적으로 많이 쓰는 형식은:

* FP32
* FP16
* BF16
* INT8

같은 것들이야.

문제는 **큰 bit-width를 쓰면 memory access와 연산 비용이 크다**는 거야.

---

## 5-2. Quantization의 기본 개념

Quantization은 연속적이거나 큰 값의 집합을 **더 작은 discrete set으로 제한하는 과정**이야. Lecture 5에서도 quantization을 “continuous or large set of values를 discrete set으로 제한하는 과정”이라고 정의해. 

쉽게 말하면:

> 원래는 다양한 실수 값을 가질 수 있었는데, 이제 몇 개의 대표값으로만 표현하자.

예를 들어 원래 weight가 이렇게 있다고 해보자.

$$
2.09,\quad -0.98,\quad 1.48,\quad 0.09
$$

이 값들을 전부 비슷한 대표값으로 바꾸면 저장이 쉬워져.

예를 들어 아주 거칠게:

$$
2.09,\ 2.12,\ 1.92,\ 1.87 \rightarrow 2.0
$$

처럼 만들 수 있어.

물론 이렇게 하면 원래 값과 바뀐 값 사이에 차이가 생겨.
이 차이를 **quantization error**라고 해.

$$\text{quantization error} = \text{original value} - \text{quantized value}$$

---

## 5-3. Data type 복습

Quantization을 이해하려면 숫자를 컴퓨터가 어떻게 표현하는지 알아야 해.

Lecture 5에서는 integer, fixed-point, floating-point를 복습해.

---

### 5-3-1. Integer

Integer는 정수 표현이야.

### Unsigned integer - 설명 생략

### Signed integer

부호가 있는 정수야.

일반적으로 2’s complement를 사용하면 $n$-bit signed integer의 범위는:

$$
-2^{n-1} \sim 2^{n-1}-1
$$

예를 들어 8-bit signed integer면:

$$
-128 \sim 127
$$

---

### 5-3-2. Fixed-point number

Fixed-point는 소수점 위치가 고정된 표현이야.

예를 들어 어떤 bit들은 정수부, 어떤 bit들은 소수부로 정해놓고 해석해.

장점은 integer 연산처럼 비교적 단순하게 처리할 수 있다는 것.
단점은 표현 가능한 range와 precision이 고정되어 있어서 유연성이 떨어질 수 있다는 것.

---

### 5-3-3. Floating-point number

Floating-point는 부동소수점 표현이야.

![14](image-14.png)

$$
(-1)^{\text{sign}}
\times
(1+\text{fraction})
\times
2^{\text{exponent}-\text{bias}}
$$

여기서 중요한 건:

* exponent bit 수가 크면 **표현 범위 range**가 커진다.
* fraction bit 수가 크면 **정밀도 precision**이 좋아진다.

---

## 5-4. Floating-point 형식 비교

Lecture 5에서는 FP32, FP16, BF16 같은 형식을 비교해.

| Type | Exponent | Fraction | Total |
| ---- | -------: | -------: | ----: |
| FP32 |        8 |       23 |    32 |
| FP16 |        5 |       10 |    16 |
| BF16 |        8 |        7 |    16 |

여기서 BF16이 중요한 이유는 FP16과 같은 16-bit지만 exponent가 8bit라서 FP32와 range가 비슷해.

즉 BF16은 precision은 FP32보다 낮지만, 큰 값이나 작은 값을 표현하는 range는 비교적 넓어.

---

## 5-5. Quantization의 큰 분류

Lecture 5에서 다루는 대표적인 quantization 방법은 크게 두 가지야.

![15](image-15.png)

1. **K-means-based quantization**
2. **Linear quantization**

그리고 마지막에 아주 극단적인 형태인 **binary / ternary quantization**을 다뤄.

---

## 5-6. K-means-based weight quantization

K-means quantization은 weight 값들을 clustering해서, 각 weight를 cluster centroid로 대체하는 방식이야.

![16](image-16.png)

weight들을 $k$개의 cluster로 나눠.

예를 들어 $k=4$이면 centroid가 4개 생겨.

$$
c_1,\ c_2,\ c_3,\ c_4
$$

각 weight는 자기와 가장 가까운 centroid로 대체돼.

---

### 5-6-1. 저장 방식

K-means quantization에서는 실제 weight 값을 전부 저장하지 않아.

![17](image-17.png)

대신 두 가지를 저장해.

1. **Codebook**: centroid 값들
2. **Index**: 각 weight가 어떤 centroid를 사용하는지

예를 들어 centroid가 4개면 index는 2bit면 충분해.

$$
4 = 2^2
$$

즉 원래 FP32 weight 하나를 32bit로 저장했다면, 이제는 각 weight마다 2bit index만 저장하면 돼.
centroid 값들은 따로 codebook에 저장하고.

Lecture 5의 K-means quantization 예시에서도 weights를 cluster index와 centroid codebook으로 나누어 저장해서 storage를 줄이는 구조를 보여줘. 

---

## 6-2. 장점과 단점

K-means quantization의 장점은 weight 분포에 맞춰 centroid를 학습하니까, 같은 bit 수에서도 비교적 quantization error를 줄일 수 있다는 거야.

하지만 단점도 있어.

실제 연산할 때 index를 보고 codebook에서 값을 lookup해야 해.
그래서 hardware에서 바로 integer 연산으로 빠르게 처리하기 어렵다.

즉 memory는 줄일 수 있지만, 실제 latency가 꼭 줄어드는 건 아니야.

이 점은 뒤의 시스템 강의에서 “model optimization이 system speedup을 보장하지 않는다”는 내용과도 연결돼.

---

# 7. Linear quantization

Linear quantization은 실수 값을 일정한 간격의 integer grid에 mapping하는 방식이야.

핵심 식은 이거야.

$$
r = S(q - Z)
$$

여기서:

| 기호  | 의미                      |
| --- | ----------------------- |
| $r$ | real value, 원래 실수 값     |
| $q$ | quantized integer value |
| $S$ | scale                   |
| $Z$ | zero-point              |

즉 integer $q$를 scale $S$와 zero-point $Z$를 이용해 실수 $r$로 복원하는 방식이야.

반대로 실수 $r$을 integer로 quantize할 때는 대략 이렇게 생각할 수 있어.

$$
q = \text{round}\left(\frac{r}{S} + Z\right)
$$

그리고 실제 integer 범위를 넘어가면 clipping을 해.

$$
q = \text{clip}
\left(
\text{round}\left(\frac{r}{S} + Z\right),
q_{\min},
q_{\max}
\right)
$$

---

## 7-1. Scale $S$

Scale은 integer 한 칸이 real value에서 얼마나 큰 간격을 의미하는지 나타내.

예를 들어 $S=0.1$이면 integer가 1 증가할 때 real value는 0.1 증가해.

$$
q = 3 \Rightarrow r = 0.3
$$

$$
q = 4 \Rightarrow r = 0.4
$$

물론 zero-point가 있으면 그만큼 이동해.

---

## 7-2. Zero-point $Z$

Zero-point는 real value 0이 integer 상에서 어디에 해당하는지 나타내는 값이야.

즉 $r=0$이 되려면:

$$
0 = S(q - Z)
$$

따라서:

$$
q = Z
$$

그래서 $Z$는 integer domain에서 real zero의 위치야.

이게 중요한 이유는 neural network에서 0이 정확히 표현되어야 하는 경우가 많기 때문이야.

---

# 8. Symmetric vs Asymmetric quantization

Linear quantization은 보통 symmetric과 asymmetric으로 나눌 수 있어.

## 8-1. Symmetric quantization

Symmetric quantization은 0을 중심으로 양쪽 범위를 대칭으로 잡는 방식이야.

보통 zero-point를 0으로 둬.

$$
Z = 0
$$

그러면 식이 단순해져.

$$
r = S q
$$

장점은 계산이 간단하다는 것.

단점은 실제 값의 분포가 대칭이 아니면 range를 낭비할 수 있다는 것.

---

## 8-2. Asymmetric quantization

Asymmetric quantization은 값의 최소/최대 범위에 맞춰 zero-point를 조정하는 방식이야.

즉 real value의 range가 예를 들어 $[r_{\min}, r_{\max}]$라면, 이 범위를 integer range $[q_{\min}, q_{\max}]$에 맞춘다.

장점은 비대칭 분포를 더 잘 표현할 수 있다는 것.

단점은 zero-point를 고려해야 해서 계산이 조금 복잡해질 수 있다는 것.

---

# 9. Bit-width와 표현 가능한 값

Quantization에서 bit-width가 작아질수록 표현 가능한 값의 개수가 줄어.

예를 들어 signed integer라고 하면:

| Bit-width | 표현 가능한 값 개수 | 예시 범위           |
| --------- | ----------: | --------------- |
| 8-bit     |        256개 | $-128 \sim 127$ |
| 4-bit     |         16개 | $-8 \sim 7$     |
| 2-bit     |          4개 | $-2 \sim 1$     |

bit-width가 줄어들면 memory는 줄지만 quantization error는 커질 수 있어.

즉 trade-off는 이거야.

> bit-width 감소 → memory/compute 효율 증가
> bit-width 감소 → quantization error 증가 가능

---

# 10. Binary quantization

Binary quantization은 weight를 두 값만으로 표현하는 극단적인 방법이야.

보통 이런 식이야.

$$
w \in {-1, +1}
$$

또는 scale을 붙이면:

$$
w \approx \alpha \cdot \text{sign}(w)
$$

여기서 $\alpha$는 scale factor야.

장점은 엄청나게 작고 빠를 수 있다는 거야.
곱셈도 거의 sign operation처럼 바꿀 수 있어.

단점은 표현력이 너무 줄어들어서 accuracy가 크게 떨어질 수 있어.

---

# 11. Ternary quantization

Ternary quantization은 세 값만 사용하는 방식이야.

$$
w \in {-1, 0, +1}
$$

또는 scale을 붙여서:

$$
w \in {-\alpha, 0, +\alpha}
$$

Binary보다 좋은 점은 0을 표현할 수 있다는 거야.

즉 weight를 아예 꺼버리는 sparsity 효과도 생겨.

하지만 여전히 표현 가능한 값이 3개뿐이라서, 일반적인 INT8 quantization보다 accuracy 손실이 클 수 있어.

---

# 12. K-means quantization vs Linear quantization

둘을 비교하면 이렇게 볼 수 있어.

| 구분    | K-means quantization | Linear quantization        |
| ----- | -------------------- | -------------------------- |
| 대표값   | cluster centroid     | 일정 간격 grid                 |
| 저장    | index + codebook     | integer + scale/zero-point |
| 장점    | weight 분포에 잘 맞출 수 있음 | hardware 친화적               |
| 단점    | lookup 필요, 연산 복잡     | 분포에 따라 error 커질 수 있음       |
| 실제 가속 | 어려울 수 있음             | INT8 연산 등으로 가속 쉬움          |

실전 inference에서는 보통 linear quantization이 hardware에서 지원되기 쉬워서 많이 쓰여.

---

# 13. Lecture 5의 핵심 흐름

Lecture 5를 하나의 흐름으로 보면 이거야.

1. 딥러닝은 memory와 연산 비용이 크다.
2. bit-width를 줄이면 memory와 energy를 줄일 수 있다.
3. Quantization은 실수 값을 작은 discrete set으로 바꾸는 것이다.
4. 이때 원래 값과 quantized 값의 차이가 quantization error다.
5. K-means quantization은 centroid와 index로 weight를 표현한다.
6. Linear quantization은 scale과 zero-point로 integer와 real value를 mapping한다.
7. Binary/Ternary quantization은 아주 낮은 bit-width의 극단적 형태다.

---

# 14. 시험/과제용 핵심 정리

| 개념                      | 핵심                                   |
| ----------------------- | ------------------------------------ |
| Quantization            | continuous value를 discrete value로 제한 |
| Quantization error      | 원래 값과 quantized 값의 차이                |
| 목적                      | memory 감소, energy 감소, low-bit 연산 활용  |
| K-means quantization    | weight를 cluster centroid로 대체         |
| Codebook                | centroid 값 저장                        |
| Index                   | 각 weight가 어떤 centroid를 쓰는지 저장        |
| Linear quantization     | $r = S(q - Z)$                       |
| Scale $S$               | integer 간격이 real value에서 의미하는 크기     |
| Zero-point $Z$          | real zero가 integer에서 대응되는 위치         |
| Symmetric quantization  | $Z=0$, 0 중심 대칭                       |
| Asymmetric quantization | min/max range에 맞춰 zero-point 사용      |
| Binary quantization     | weight를 ${-1,+1}$로 제한                |
| Ternary quantization    | weight를 ${-1,0,+1}$로 제한              |

---

한 문장으로 정리하면:

> Lecture 5는 neural network quantization의 기본 강의로, 실수 weight/activation을 더 적은 bit의 discrete 값으로 표현해 memory와 연산 비용을 줄이는 방법을 배우며, 핵심은 K-means quantization과 linear quantization, 그리고 scale/zero-point 개념을 이해하는 것이다.







<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true
    },
    "HTML-CSS": { linebreaks: { automatic: true } }
  });
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


