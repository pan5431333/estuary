package estuary.concurrency

import akka.actor.ActorSystem
import com.typesafe.config.ConfigFactory

object createWorkerActorSystem {
  def main(args: Array[String]): Unit = {
    val systemName = if (args == null) "Worker" else args(0)

    ActorSystem(systemName, ConfigFactory.load("WorkerActorSystem"))
  }
}