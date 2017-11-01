package estuary.concurrency.decentralized

import akka.actor.{Actor, ActorRef, Props}
import estuary.concurrency.MyMessage
import estuary.concurrency.decentralized.DecentralizedBatchCalculator.{CurrentParam, GetParam, TrainingDone}
import estuary.concurrency.decentralized.Manager.{CostHistory, GetCostHistory, StartTrain}


/**
  * Created by mengpan on 2017/10/28.
  */
class Manager(workers: Seq[ActorRef]) extends Actor{

  private[this] var nWorkActorsDone: Int = 0
  private[this] val nWorkActors: Int = workers.length
  private[this] var appMainActor: ActorRef = _
  private[this] var costHistory: List[Double] = List()

  override def receive: Receive = {
    case StartTrain =>
      appMainActor = sender
      workers.foreach(_ ! StartTrain)

    case TrainingDone =>
      nWorkActorsDone += 1
      if (nWorkActorsDone >= nWorkActors)
        workers.head ! GetParam

    case CostHistory(cost) => costHistory = cost :: costHistory

    case CurrentParam(param) => appMainActor ! CurrentParam(param)

    case GetCostHistory => sender ! costHistory.reverse
  }
}

object Manager {

  def props(workers: Seq[ActorRef]): Props = {
    Props(classOf[Manager], workers)
  }

  sealed trait ManagerMsg extends MyMessage
  final case object StartTrain extends ManagerMsg
  final case class CurrentParams(params: AnyRef) extends ManagerMsg
  final case object GetCostHistory extends ManagerMsg
  final case class CostHistory(cost: Double) extends ManagerMsg
}
