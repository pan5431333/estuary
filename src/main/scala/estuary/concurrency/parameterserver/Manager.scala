package estuary.concurrency.parameterserver

import akka.actor.{Actor, ActorRef, Props}
import estuary.concurrency.parameterserver.BatchGradCalculatorActor.{StartTrain, TrainingDone}
import estuary.concurrency.parameterserver.ParameterServerActor.{CostHistory, CurrentParams, GetCostHistory, GetCurrentParams}

class Manager(parameterServer: ActorRef, workers: Seq[ActorRef]) extends Actor {

  private[this] var nWorkActorsDone: Int = 0
  private[this] val nWorkActors: Int = workers.length
  private[this] var appMainActor: ActorRef = _
  private[this] var costHistory: List[Double] = List()

  override def receive = {
    case StartTrain =>
      appMainActor = sender
      workers foreach (_ ! StartTrain)
      context.watch(parameterServer)
      workers foreach context.watch

    case TrainingDone =>
      nWorkActorsDone += 1
      if (nWorkActorsDone >= nWorkActors) parameterServer ! GetCurrentParams

    case CurrentParams(params) => appMainActor ! CurrentParams(params)

    case CostHistory(cost) => costHistory = cost :: costHistory

    case GetCostHistory => sender ! costHistory.reverse
  }

}

object Manager {
  def props(parameterServer: ActorRef, workers: Seq[ActorRef]): Props = {
    Props(classOf[Manager], parameterServer, workers)
  }
}
