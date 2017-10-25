package estuary

import java.io.{File, FileNotFoundException}

import akka.actor.ActorSystem
import com.typesafe.config.{Config, ConfigFactory}

package object concurrency {
  def createActorSystem(name: String, configFilePath: String): ActorSystem = {
    checkFileExist(configFilePath)
    ActorSystem(name, ConfigFactory.load(configFilePath))
  }

  def createActorSystem(name: String, ip: String, port: Int): ActorSystem = {
    ActorSystem(name, getDefaultConfig(ip, port))
  }

  def createActorSystem(name: String): ActorSystem = {
    createActorSystem(name, "127.0.0.1", 2552)
  }

  @throws[FileNotFoundException]
  def checkFileExist(str: String): Unit = {
    val file = new File(str)
    if (!file.exists()) throw new FileNotFoundException(s"file $str does not exist")
  }

  private def getDefaultConfig(ip: String, port: Int): Config = {
    val configStr =
      s"""
        |akka {
        |  actor {
        |    provider = "akka.remote.RemoteActorRefProvider"
        |  }
        |
        |  remote {
        |    maximum-payload-bytes = 30000000 bytes
        |    netty.tcp {
        |      hostname = $ip
        |      port = $port
        |      message-frame-size =  30000000b
        |      send-buffer-size =  30000000b
        |      receive-buffer-size =  30000000b
        |      maximum-frame-size = 30000000b
        |    }
        |  }
        |}
      """.stripMargin
    ConfigFactory.parseString(configStr)
  }
}
