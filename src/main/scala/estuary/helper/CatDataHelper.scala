package estuary.helper


import estuary.data.Cat
import org.apache.log4j.Logger

import scala.io.Source

/**
  * Created by mengpan on 2017/8/15.
  */
object CatDataHelper {
  val logger: Logger  = Logger.getLogger(this.getClass)


  def getAllCatData: DlCollection[Cat] = {
    val labels = getLabels
    val catNonCatLabels = getBalancedCatNonCatLabels(labels)

    logger.info("Reading files cocurrently...")
    val catList = catNonCatLabels.par.map { indexedLabel =>

      val fileNumber = indexedLabel._1
      val label = indexedLabel._2
      val animalFileName: String = "/Users/mengpan/Downloads/train/" + fileNumber + ".png"
      val feature = ImageReader.readImageToRGBVector(animalFileName)

      (feature, label)
    }
      .filter(_._1.isDefined)
      .map(f => Cat(f._1.get, f._2))
      .toList

    new DlCollection[Cat](catList)
  }

  private def getLabels: Vector[(Int, String)] = {
    Source
      .fromFile("/Users/mengpan/Downloads/trainLabels.csv")
      .getLines()
      .map { eachRow =>
        val split = eachRow.split(",")
        (split(0), split(1))
      }
      .filter { eachRow =>
        eachRow._1 != "id"
      }
      .map { eachRow =>
        (eachRow._1.toInt, eachRow._2)
      }
      .toVector
  }

  private def getBalancedCatNonCatLabels(labels: Vector[(Int, String)]): Vector[(Int, Int)] = {
    labels
      .map { label =>
        val numLabel = label._2 match {
          case "cat" => 1
          case "frog" => 2
          case "truck" => 3
          case "automobile" => 4
          case "deer" => 5
//          case "bird" => 6
//          case "horse" => 7
//          case "ship" => 8
//          case "airplane" => 9
//          case "dog" => 0
          case _ => 0
        }
        (label._1, numLabel)
      }.filter( _._2 != 0)
  }

}
