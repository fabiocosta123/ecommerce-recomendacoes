# Recomendação de produtos com rede neural

Loja de estudo em que o navegador treina uma rede neural para sugerir o próximo produto de cada pessoa, a partir do que ela e os outros clientes já compraram.

O catálogo, os perfis e a tela já estão prontos. O modelo entra em `src/workers/modelTrainingWorker.js`: é ali que a rede é criada, treinada e consultada.

## O problema

Um e-commerce conhece o catálogo e o histórico de cada cliente. A pergunta da aula é: dado esse histórico, quais produtos essa pessoa ainda não comprou e tem mais chance de querer?

Os dados de exemplo já mostram o padrão. Quem compra fone e relógio costuma estar no grupo de eletrônicos. Quem compra camiseta e calça está no grupo de vestuário. A rede precisa aprender esses agrupamentos sozinha, em vez de usar uma regra escrita à mão.

## Como a rede resolve isso

O treino usa os usuários de `data/users.json`. Cada exemplo vira um vetor numérico: idade e um sinal de quais produtos (ou categorias) aquela pessoa já comprou. A saída é uma nota para cada item do catálogo. Nota alta significa “essa pessoa provavelmente compraria isto”.

Durante o treino, o TensorFlow.js ajusta os pesos para reduzir o erro entre a nota prevista e o que de fato foi comprado. Cada passagem completa pelos exemplos é uma época. O `tfjs-vis` desenha duas curvas nessa etapa:

- **erro (loss):** o quanto a previsão ainda erra
- **precisão (accuracy):** o quanto a previsão acerta

Na recomendação, o modelo recebe só o usuário escolhido na tela e devolve os produtos com maior nota, deixando de fora o que ele já comprou.

O cálculo roda num [Web Worker](https://developer.mozilla.org/docs/Web/API/Web_Workers_API), fora da thread da página, para o treino não travar os botões e a lista de produtos.

## O que a tela já faz

1. Carrega usuários e produtos dos JSON em `data/`.
2. Mostra idade e compras de quem for selecionado.
3. Registra uma compra nova no `sessionStorage` ao clicar em comprar.
4. **Train Model** manda a lista de usuários para o worker.
5. **Run Recommendation** pede sugestões para o usuário atual, depois que o treino termina.
6. Abre o visor do TensorFlow.js com as curvas de treino.

O botão de recomendação só é liberado com um usuário selecionado e um treino concluído.

Hoje `trainModel` e `recommend` no worker apenas simulam progresso. A interface, os eventos e o visor já estão ligados; falta a rede de verdade.

## Dados

`data/products.json` — dez produtos, com `id`, `name`, `category`, `price` e `color`. As categorias são eletrônicos, vestuário, calçados e acessórios.

`data/users.json` — cinco clientes, com `id`, `name`, `age` e `purchases`. Cada compra repete o produto comprado.

Compras feitas na tela ficam só na sessão do navegador. Recarregar a página volta ao JSON original.

## Onde cada parte mora

| Pasta | Papel |
| --- | --- |
| `data/` | Catálogo e histórico usados no treino |
| `src/view/` | Tela: perfil, produtos, botões de treino e visor |
| `src/controller/` | Liga a tela aos serviços e ao worker |
| `src/service/` | Lê os JSON e guarda usuários no `sessionStorage` |
| `src/events/` | Avisos da página: usuário escolhido, treino, recomendação |
| `src/workers/modelTrainingWorker.js` | TensorFlow.js: criar a rede, treinar e recomendar |

O worker escuta duas ações:

- `train:model` — recebe `{ users }` e deve treinar
- `recommend` — recebe o usuário atual e deve responder com a lista de produtos sugeridos

Quando o treino avança, o worker publica `progress:update`, `training:log` (época, loss e accuracy) e, no fim, `training:complete`. A recomendação volta como `recommend`, e a lista de produtos da página é trocada pelas sugestões.

## Como rodar

Na pasta do projeto:

```sh
npm install
npm start
```

Abra `http://localhost:3000`. Escolha um usuário, clique em **Train Model** e, ao terminar, em **Run Recommendation**.
