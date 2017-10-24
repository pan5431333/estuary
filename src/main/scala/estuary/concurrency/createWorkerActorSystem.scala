package estuary.concurrency

import akka.actor.ActorSystem
import com.typesafe.config.ConfigFactory

object createWorkerActorSystem {
  def main(args: Array[String]): Unit = {
    val systemName = if (args == null) "Worker" else args(0)
    val configPath = if (args.length > 1) args(1) else "WorkerActorSystem"

    ActorSystem(systemName, ConfigFactory.load(configPath))
  }
}