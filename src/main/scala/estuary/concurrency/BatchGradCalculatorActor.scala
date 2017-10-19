package estuary.concurrency

import akka.actor.{Actor, ActorRef}
import breeze.linalg.DenseMatrix
import estuary.components.optimizer.AbstractAkkaDistributed.CostHistory
import estuary.components.optimizer.Distributed
import estuary.concurrency.BatchGradCalculatorActor.Start
import estuary.concurrency.ParameterServerActor._
import estuary.model.Model
import org.apache.log4j.Logger

/**
  *
  * @tparam M Model Parameter
  * @tparam O Optimizer Parameter
  */
class BatchGradCalculatorActor[M, O](feature: DenseMatrix[Double],
                                     label: DenseMatrix[Double],
                                     model: Model[M],
                                     iteration: Int,
                                     shuffleFunc: (DenseMatrix[Double], DenseMatrix[Double]) => Iterator[(DenseMatrix[Double], DenseMatrix[Double])],
                                     updateFunc: (O, M, Int) => O,
                                     parameterServer: ActorRef,
                                     convertFunc: O => M)
  extends Actor {
  private[this] val logger = Logger.getLogger(this.getClass)
  private[this] var shuffledData: Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = shuffleFunc(feature, label)
  private[this] var miniBatchIndex: Int = 0
  private[this] var iterTime: Int = 0
  private[this] var currentFeature: DenseMatrix[Double] = _
  private[this] var currentLabel: DenseMatrix[Double] = _
  private[this] var grads: M = _
  private[this] lazy val miniBatchSize: Int = currentFeature.rows
  private[this] var mainSender: ActorRef = _

  override def receive: Actor.Receive = {
    case Start =>
      parameterServer ! GetCurrentParams
      mainSender = sender()

    case CurrentParams(params) =>
      iter(params.asInstanceOf[O])
      if (iterTime < iteration) {
        parameterServer ! GetCurrentParamsForUpdate
        parameterServer ! GetCurrentParams
      } else context.stop(self)

    case CurrentParamsForUpdate(params, miniBatchTime) =>
      parameterServer ! UpdateParams(updateFunc(params.asInstanceOf[O], grads, miniBatchIndex))
  }

  private def iter(params: O): Unit = {
    val cost = calculateCost(convertFunc(params))
    val printCostUnit = math.max(currentFeature.rows / this.miniBatchSize / 5, 10)
    Distributed.printCostInfo(cost, iterTime, miniBatchIndex, printCostUnit, logger)
    if (miniBatchIndex % printCostUnit == 0) mainSender ! CostHistory(cost)
    grads = calculateGrads(convertFunc(params))
  }

  private def calculateCost(params: M): Double = {
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

  private def calculateGrads(params: M): M = {
    model.backward(currentLabel, params)
  }
}

object BatchGradCalculatorActor {

  sealed trait BatchGradCalculatorActorMsg

  case object Start extends BatchGradCalculatorActorMsg


}


