package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import org.apache.log4j.Logger

import scala.collection.mutable


/**
  * Optimizer Interface, all of whose implementations
  * MUST implement abstract method: "optimize"
  *
  * @note This optimizer can be ONLY used for optimizing Machine Learning-like
  *       problems, which means that input-output (i.e. feature-label) data are needed.
  *       NOT for general optimization or mathematical planning problem.
  */
trait Optimizer {
  val logger: Logger = Logger.getLogger(this.getClass)

  var iteration: Int = _
  var learningRate: Double = _

  def setIteration(iteration: Int): this.type = {
    this.iteration = iteration
    this
  }

  def setLearningRate(learningRate: Double): this.type = {
    assert(learningRate > 0, "Nonpositive learning rate: " + learningRate)
    this.learningRate = learningRate
    this
  }

  /** Storing cost history after every iteration. */
  val costHistory: mutable.MutableList[Double] = new mutable.MutableList[Double]()

  /**
    * Optimizing Machine Learning-like models' parameters on a training dataset (feature, label).
    *
    * @param feature      DenseMatrix of shape (n, p) where n: the number of
    *                     training examples, p: the dimension of input feature.
    * @param label        DenseMatrix of shape (n, q) where n: the number of
    *                     training examples, q: number of distinct labels.
    * @param initParams   Initialized parameters.
    * @param forwardFunc  The cost function.
    *                     inputs: (feature, label, params) of type (DenseMatrix[Double], DenseMatrix[Double], T)
    *                     output: cost of type Double.
    * @param backwardFunc A function calculating gradients of all parameters.
    *                     input: (label, params) of type (DenseMatrix[Double], T)
    *                     output: gradients of params of type T.
    * @return Trained parameters.
    */
  def optimize(feature: DenseMatrix[Double], label: DenseMatrix[Double])
              (initParams: Seq[DenseMatrix[Double]])
              (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Double)
              (backwardFunc: (DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]]

}
