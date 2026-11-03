import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

const makeContext = (product, users) => {
    const ages = users.map(u => u.age);
    const prices = product.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(product.map(p => p.color))];
    const categories = [...new Set(product.map(p => p.category))];

    const colorIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index];
        })
    )

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index];
        })
    )

    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        product.map((product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] : midAge;

            return [product.name, normalize(avg, minAge, maxAge)];
        }))
    )

    return {
        product,
        users,
        colorIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        productAvgAgeNorm,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + colors.length + categories.length
    }

}

const oneHotWeighted = (index, length, weight) => {
    return tf.oneHot(index, length).cast('float32').mul(weight);
}
function encondeProduct(product, context) {
    // normalizando dados para ficar de 0 a 1 e aplicar peso da recomendação
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const categories = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const colors = oneHotWeighted(
        context.colorIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d([price, age, colors, categories]);
}

// retorna o perfil de compra de cada usuario
function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => encondeProduct(product, context))
        )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado
            tf.tensor1d(
                [
                    normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
                ]
            ),
            tf.zeros([context.numColors]), // cor ignorada
            tf.zeros([context.numCategories]), // categoria ignorada
        ]
    ).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];

    context.users.filter(u => u.purchases.length).forEach(user => {
        const encodedUser = encodeUser(user, context);
        if (!encodedUser) return;

        const userVector = encodedUser.dataSync();
        context.product.forEach(product => {
            const productVector = encondeProduct(product, context).dataSync();
            const label = user.purchases.some(purchase => purchase.name === product.name) ? 1 : 0;

            // combinar usuario + produto
            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        })
    })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        // tamanho = userVector + productVector
        inputDimention: context.dimentions * 2
    }
}

async function configureNeuraNetAndTrain(trainData) {
    const model = tf.sequential();

    // camada de entrada
    // inputShape:  numero de feature por exemplo de treino(trainData.inputDim)
    // exemplo: Se o vetor do produto + usuario = 20 numeros, então inputDim é 20
    // units: 128 neuronios (muitos 'olhos' para detectar padrões)
    // activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )

    // camada oculta 1
    // 64 neurônnios (menos que a primeira camada: começa a comprimir informação)
    // activation: 'relu' (ainda extrai combinações relevantes da features)
    // como 'relu' retorna apenas numeros positivos (informações interessantes)
    // o fato da segunda camada ter menos neurônios não significa que é menos importante, mas sim que está
    // trabalhando apenas com informações já filtradas da primeira camada
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // camada oculta 2
    // 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    // exemplo: de muitos sinais, mantém apenas os padrões mais fortes
    // activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )

    // camada de saida
    // 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // activation: 'sigmoid' comprime o resultado para o intervalo 0 e 1
    // exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({
            units: 1, activation: 'sigmoid'
        })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const product = await fetch('/data/products.json').then(res => res.json());

    const context = makeContext(product, users);
    context.productVectors = product.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encondeProduct(product, context).dataSync()
        }
    })



    _globalCtx = context;

    const trainData = createTrainingData(context);
    _model = await configureNeuraNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user) {
    if (!_model) return;
    const context = _globalCtx;

    const userVector = encodeUser(user, _globalCtx).dataSync();
    /*
        Em aplicações reais:
        Armazene todos os vetores de produtos em um banco de dados vetorial como Postgres, Neo4j ou Pinecone
        Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
        Execute _model.predict() apenas nesses produtos

        2 - Crie pares de entrada: para cada produto, concatene o vetor do usuário
        com o vetor codificado do produto
        Por que? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto)
     */
    const inputs = context.productVectors.map(({vector}) => {
        return [...userVector, ...vector]
    });    

    /*
        3 - Converta todos esses pares (usuário, produto) em um único tensor  
        Formato: [numProdutos, inputDim] 
    */
    const inputTensor = tf.tensor2d(inputs);

    /*
        4 - Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
        O resultado é uma pontuação para cada produto entre 0 e 1
        Quanto maior, maior a probabilidade do usário querer aquele produto 
    */
    const predictions = _model.predict(inputTensor);

    /*
        5 - Extraia as pontuações para um array JS normal
    */
    const scores = predictions.dataSync();

    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            // previsão do modelo para este produto
            score: scores[index]
        }
    })

    const sortedItems = recommendations.sort((a, b) => b.score - a.score)

    // envia a lista ordenada de produtos recomendados
    // para a thread principal (a UI pode exibi -los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });    
}



const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
