package estuary.helper

import breeze.linalg.DenseMatrix
import estuary.data.GasCensor
import org.apache.log4j.Logger

import scala.io.Source

/**
  * Created by mengpan on 2017/8/24.
  */
object GasCensorDataHelper {
  val logger: Logger = Logger.getLogger("GasCensorDataHelper")

  def getAllData(dataDir: String): DlCollection[GasCensor] = {
    val directoryPath = dataDir

    val gasCensorSeq = (1 to 10).par.flatMap { i =>
      val filePath = directoryPath + "batch" + i + ".dat"

      Source.fromFile(filePath)
        .getLines()
        .flatMap { eachLine =>
          val words = eachLine.split(" ")
          val keyValuePair = words.map { eachWord =>
            if (!eachWord.contains(":")) ("label", eachWord)
            else {
              val colFeature = eachWord.split(":")
              (colFeature(0), colFeature(1))
            }
          }
          keyValuePair
        }
    }

    val labels = gasCensorSeq
      .filter { eachLine =>
        eachLine._1 == "label"
      }
      .map(_._2.toDouble)
      .toVector

    val numExamples = labels.length

    val featuresArr = gasCensorSeq
      .filter { eachLine =>
        eachLine._1 != "label"
      }
      .map(_._2.toDouble)
      .toArray

    val featureMatrix = new DenseMatrix[Double](128, numExamples, featuresArr).t

    val gasCensorList = (0 until numExamples).map { i =>
      GasCensor(featureMatrix(i, ::).t, labels(i))
    }
      .filter { each =>
        each.label == 1.0 || each.label == 2.0
      }
      .map { each =>
        GasCensor(each.feature, if (each.label == 1) 1.0 else 0.0)
      }
      .toList


    val numPositives = gasCensorList
      .count {
        _.label == 1.0
      }

    val numNegatives = gasCensorList
      .count {
        _.label == 0.0
      }

    logger.info("数据读取完毕，正样本数量：" + numPositives + "|负样本数量：" + numNegatives)

    new DlCollection[GasCensor](gasCensorList)
  }

}
