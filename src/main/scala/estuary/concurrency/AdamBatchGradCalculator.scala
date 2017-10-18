package estuary.concurrency

import akka.actor.{Actor, ActorRef}
import breeze.linalg.DenseMatrix
import estuary.components.optimizer.AkkaAdamOptimizer.CostHistory
import estuary.components.optimizer.Distributed
import estuary.concurrency.AdamBatchGradCalculator._
import estuary.concurrency.ParameterServer._
import estuary.model.Model
import org.apache.log4j.Logger

/**
  *
  * @tparam A Model Parameter
  * @tparam B Optimizer Parameter
  */
class AdamBatchGradCalculator[A, B](feature: DenseMatrix[Double],
                                    label: DenseMatrix[Double],
                                    model: Model[A],
                                    iteration: Int,
                                    shuffleFunc: (DenseMatrix[Double], DenseMatrix[Double]) => Iterator[(DenseMatrix[Double], DenseMatrix[Double])],
                                    updateFunc: (B, A, Int) => B,
                                    parameterServer: ActorRef,
                                    convertFunc: B => A = (a: B) => a.asInstanceOf[A])
  extends Actor {
  private[this] val logger = Logger.getLogger(this.getClass)
  private[this] var shuffledData: Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = shuffleFunc(feature, label)
  private[this] var miniBatchIndex: Int = 0
  private[this] var iterTime: Int = 0
  private[this] var currentFeature: DenseMatrix[Double] = _
  private[this] var currentLabel: DenseMatrix[Double] = _
  private[this] var grads: A = _
  private[this] lazy val miniBatchSize: Int = currentFeature.rows
  private[this] var mainSender: ActorRef = _

  override def receive: Actor.Receive = {
    case Start() =>
      parameterServer ! GetCurrentParams
      mainSender = sender()

    case CurrentParams(params) =>
      iter(params.asInstanceOf[B])
      if (iterTime < iteration) {
        parameterServer ! GetCurrentParamsForUpdate
        parameterServer ! GetCurrentParams
      } else context.stop(self)

    case CurrentParamsForUpdate(params, miniBatchTime) =>
      parameterServer ! UpdateParams(updateFunc(params.asInstanceOf[B], grads, miniBatchIndex))
  }

  private def iter(params: B): Unit = {
    val cost = calculateCost(convertFunc(params))
    val printCostUnit = math.max(currentFeature.rows / this.miniBatchSize / 5, 10)
    Distributed.printCostInfo(cost, iterTime, miniBatchIndex, printCostUnit, logger)
    if (miniBatchIndex % printCostUnit == 0) mainSender ! CostHistory(cost)
    grads = calculateGrads(convertFunc(params))
  }

  private def calculateCost(params: A): Double = {
    if (shuffledData.hasNext) {
      val (onefeature, onelabel) = shuffledData.next()
      currentFeature = onefeature
      currentLabel = onelabel
      miniBatchIndex += 1
    } else {
      shuffledData = shuffleFunc(feature, label)
      miniBatchIndex = 0
      iterTime += 1
      val (onefeature, onelabel) = shuffledData.next()
      currentFeature = onefeature
      currentLabel = onelabel
    }
    model.forward(currentFeature, currentLabel, params)
  }

  private def calculateGrads(params: A): A = {
    model.backward(currentLabel, params)
  }
}

object AdamBatchGradCalculator {

  sealed trait AdamBatchGradCalculatorMsg

  case class Iter() extends AdamBatchGradCalculatorMsg

  case class Start() extends AdamBatchGradCalculatorMsg


}


