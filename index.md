---
layout: default
---

# doddle-model
[doddle-model](https://github.com/picnicml/doddle-model) is an in-memory machine learning library that can be summed up with three main characteristics:
* it is built on top of [Breeze](https://github.com/scalanlp/breeze)
* it provides [immutable estimators](https://en.wikipedia.org/wiki/Immutable_object) that are a _doddle_ to use in parallel code
* it exposes its functionality through a [scikit-learn](https://github.com/scikit-learn/scikit-learn)-like API [2] in idiomatic Scala using [typeclasses](https://en.wikipedia.org/wiki/Type_class)

#### How does it compare to existing solutions?
`doddle-model` takes the position of scikit-learn in Scala and as a consequence, it's much more lightweight than e.g. Spark ML. Fitted models can be deployed anywhere, from simple applications to concurrent, distributed systems built with Akka, Apache Beam or a framework of your choice. Training of estimators happens in-memory, which is advantageous unless you are dealing with enormous datasets that absolutely cannot fit into RAM.

You can chat with us [on gitter](https://gitter.im/picnicml/doddle-model).

## Installation
<a href="https://search.maven.org/search?q=g:io.github.picnicml">
    <img src="https://img.shields.io/maven-central/v/io.github.picnicml/doddle-model_2.12.svg?style=flat-square&label=maven%20central" alt="latest release"/>
</a>

The project is published for Scala versions 2.11, 2.12 and 2.13. Add the dependency to your SBT project definition:
```scala
libraryDependencies  ++= Seq(
  "io.github.picnicml" %% "doddle-model" % "<latest_version>",
  // add optionally to utilize native libraries for a significant performance boost
  "org.scalanlp" %% "breeze-natives" % "1.0"
)
```
Note that the latest version is displayed in the _maven central_ badge above and that the _v_ prefix should be removed from the SBT definition.

## Getting Started
This is a complete list of code examples, for an example of how to serve a trained [doddle-model](https://github.com/picnicml/doddle-model) in a pipeline implemented with Apache Beam see [doddle-beam-example](https://github.com/picnicml/doddle-beam-example).

#### 1. Feature Preprocessing
* [Standard Scaler](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/preprocessing/StandardScalerExample.scala)
* [Range Scaler](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/preprocessing/RangeScalerExample.scala)
* [Binarizer](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/preprocessing/BinarizerExample.scala)
* [Normalizer](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/preprocessing/NormalizerExample.scala)
* [One-Hot Encoder](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/preprocessing/OneHotEncoderExample.scala)
* [Mean Value Imputation](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/impute/MeanValueImputerExample.scala)
* [Most Frequent Value Imputation](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/impute//MostFrequentValueImputerExample.scala)

#### 2. Metrics
* [Classification Metrics](https://github.com/picnicml/doddle-model/blob/master/src/main/scala/io/picnicml/doddlemodel/metrics/ClassificationMetrics.scala)
* [Regression Metrics](https://github.com/picnicml/doddle-model/blob/master/src/main/scala/io/picnicml/doddlemodel/metrics/RegressionMetrics.scala)
* [Ranking Metrics](https://github.com/picnicml/doddle-model/blob/master/src/main/scala/io/picnicml/doddlemodel/metrics/RankingMetrics.scala)
* [ROC curve visualization](https://picnicml.github.io/doddle-model-examples/roc-curve-visualization.html)

#### 3. Baseline models
* [Most Frequent Classifier](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/dummy/MostFrequentClassifierExample.scala)
* [Stratified Classifier](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/dummy/StratifiedClassifierExample.scala)
* [Uniform Classifier](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/dummy/UniformClassifierExample.scala)
* [Mean Regressor](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/dummy/MeanRegressorExample.scala)
* [Median Regressor](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/dummy/MedianRegressorExample.scala)

#### 4. Linear models
* [Linear Regression](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/linear/LinearRegressionExample.scala)
* [Logistic Regression](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/linear/LogisticRegressionExample.scala)
* [Softmax Classifier](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/linear/SoftmaxClassifierExample.scala)
* [Poisson Regression](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/linear/PoissonRegressionExample.scala)

#### 5. Model Selection
* [K-Fold Cross-Validation](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/modelselection/KFoldExample.scala)
* [Group K-Fold Cross-Validation](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/modelselection/GroupKFoldExample.scala)
* [Grid Search](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/modelselection/GridSearchExample.scala)
* [Random Search](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/modelselection/RandomSearchExample.scala)

#### 6. Miscellaneous
* [Reading Data](https://github.com/picnicml/doddle-model-examples/wiki/Reading-CSV-Data)
* [Shuffling Data](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/misc/ShuffleDatasetExample.scala)
* [Splitting Data](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/misc/SplitDatasetExample.scala)
* [Feature Preprocessing Pipeline](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/pipeline/PipelineExample.scala)
* [Estimator Persistence](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/misc/EstimatorPersistenceExample.scala)

#### 7. Use Cases
* [Kaggle House Prices](https://github.com/picnicml/doddle-model-examples/blob/master/src/main/scala/io/picnicml/doddlemodel/examples/usecase/HousePrices.scala)

## Contributing
<a href="https://circleci.com/gh/picnicml/doddle-model">
    <img src="https://img.shields.io/circleci/project/github/picnicml/doddle-model/master.svg?style=flat-square&label=circleci" alt="build status"/>
</a>
<a href="https://app.codacy.com/project/inejc/doddle-model/dashboard">
    <img src="https://img.shields.io/codacy/coverage/9f1dad5c6d6041dd85db71adabba3e72.svg?style=flat-square&label=codacy" alt="coverage"/>
</a>
<a href="https://app.codacy.com/project/inejc/doddle-model/dashboard">
    <img src="https://img.shields.io/codacy/grade/9f1dad5c6d6041dd85db71adabba3e72/master.svg?style=flat-square&label=codacy" alt="code quality"/>
</a>

Want to help us? We have a [document](https://github.com/picnicml/doddle-model/blob/master/.github/CONTRIBUTING.md) that will make deciding how to do that much easier. Be sure to also check the [roadmap](https://github.com/picnicml/doddle-model/wiki/Roadmap).

## Performance
[doddle-model](https://github.com/picnicml/doddle-model) is developed with performance in mind.

#### 1. Native Linear Algebra Libraries
[Breeze](https://github.com/scalanlp/breeze) utilizes [netlib-java](https://github.com/fommil/netlib-java) for accessing hardware optimised linear algebra libraries (note that the `breeze-natives` dependency needs to be added to the SBT project definition). TL;DR seeing something like
```
INFO: successfully loaded /var/folders/9h/w52f2svd3jb750h890q1x4j80000gn/T/jniloader3358656786070405996netlib-native_system-osx-x86_64.jnilib
```
means that BLAS/LAPACK/ARPACK implementations are used. For more information see the [Breeze](https://github.com/scalanlp/breeze) documentation.

#### 2. Memory
If you encounter `java.lang.OutOfMemoryError: Java heap space` increase the maximum heap size with `-Xms` and `-Xmx` JVM properties. E.g. use `-Xms8192m -Xmx8192m` for initial and maximum heap space of 8Gb. Note that the maximum heap limit for the 32-bit JVM is 4Gb (at least in theory) so make sure to use 64-bit JVM if more memory is needed. If the error still occurs and you are using hyperparameter search or cross validation, see the next section.

#### 3. Parallelism
To limit the number of threads running at one time (and thus memory consumption) when doing cross validation and hyperparameter search, a `FixedThreadPool` executor is used. By default maximum number of threads is set to the number of system's cores. Set the `-DmaxNumThreads` JVM property to change that, e.g. to allow for 16 threads use `-DmaxNumThreads=16`.

## Benchmarks
All experiments ran multiple times (iterations) for all implementations and with fixed hyperparameters, selected in a way such that models yielded similar test set performance.

#### 1. Linear Regression
- dataset with 150000 training examples and 27147 test examples (10 features)
- each experiment ran for 100 iterations
- [scikit-learn code](https://github.com/picnicml/doddle-benchmark/blob/master/src/main/scala/com/picnicml/doddlemodel/linear/sklearn_linear_regression.py), [doddle-model code](https://github.com/picnicml/doddle-benchmark/blob/master/src/main/scala/com/picnicml/doddlemodel/linear/DoddleLinearRegression.scala)

<table>
<tr>
    <th>Implementation</th>
    <th>RMSE</th>
    <th>Training Time</th>
    <th>Prediction Time</th>
</tr>
<tr>
    <td>scikit-learn</td>
    <td>3.0936</td>
    <td>0.042s (+/- 0.014s)</td>
    <td>0.002s (+/- 0.002s)</td>
</tr>
<tr>
    <td>doddle-model</td>
    <td>3.0936</td>
    <td>0.053s (+/- 0.061s)</td>
    <td>0.002s (+/- 0.004s)</td>
</tr>
</table>

#### 2. Logistic Regression
- dataset with 80000 training examples and 20000 test examples (250 features)
- each experiment ran for 100 iterations
- [scikit-learn code](https://github.com/picnicml/doddle-benchmark/blob/master/src/main/scala/com/picnicml/doddlemodel/linear/sklearn_logistic_regression.py), [doddle-model code](https://github.com/picnicml/doddle-benchmark/blob/master/src/main/scala/com/picnicml/doddlemodel/linear/DoddleLogisticRegression.scala)

<table>
<tr>
    <th>Implementation</th>
    <th>Accuracy</th>
    <th>Training Time</th>
    <th>Prediction Time</th>
</tr>
<tr>
    <td>scikit-learn</td>
    <td>0.8389</td>
    <td>2.789s (+/- 0.090s)</td>
    <td>0.005s (+/- 0.006s)</td>
</tr>
<tr>
    <td>doddle-model</td>
    <td>0.8377</td>
    <td>3.080s (+/- 0.665s)</td>
    <td>0.025s (+/- 0.025s)</td>
</tr>
</table>

#### 3. Softmax Classifier
- MNIST dataset with 60000 training examples and 10000 test examples (784 features)
- each experiment ran for 50 iterations
- [scikit-learn code](https://github.com/picnicml/doddle-benchmark/blob/master/src/main/scala/com/picnicml/doddlemodel/linear/sklearn_softmax_classifier.py), [doddle-model code](https://github.com/picnicml/doddle-benchmark/blob/master/src/main/scala/com/picnicml/doddlemodel/linear/DoddleSoftmaxClassifier.scala)

<table>
<tr>
    <th>Implementation</th>
    <th>Accuracy</th>
    <th>Training Time</th>
    <th>Prediction Time</th>
</tr>
<tr>
    <td>scikit-learn</td>
    <td>0.9234</td>
    <td>21.243s (+/- 0.303s)</td>
    <td>0.074s (+/- 0.018s)</td>
</tr>
<tr>
    <td>doddle-model</td>
    <td>0.9223</td>
    <td>25.749s (+/- 1.813s)</td>
    <td>0.042s (+/- 0.032s)</td>
</tr>
</table>

## Core Maintainers
This is a collaborative project which wouldn't be possible without all the [awesome contributors](https://github.com/picnicml/doddle-model/graphs/contributors). The core team currently consists of the following developers:
- [@inejc](https://github.com/inejc)
- [@matejklemen](https://github.com/matejklemen)

## Resources
* [1] [Pattern Recognition and Machine Learning, Christopher Bishop](http://www.springer.com/gp/book/9780387310732)
* [2] [API design for machine learning software: experiences from the scikit-learn project, L. Buitinck et al.](https://arxiv.org/abs/1309.0238)
* [3] [UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science, Dua, D. and Karra Taniskidou, E.](http://archive.ics.uci.edu/ml)
