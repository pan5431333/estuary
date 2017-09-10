package org.mengpan.deeplearning.helper


import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.data.Cat

import scala.io.Source

/**
  * Created by mengpan on 2017/8/15.
  */
object CatDataHelper {
  val logger = Logger.getLogger(this.getClass)


  def getAllCatData: DlCollection[Cat] = {

    val labels = getLabels()

    val catNonCatLabels = getBalancedCatNonCatLabels(labels)

    val catList = catNonCatLabels.map{indexedLabel =>

      val fileNumber = indexedLabel._1
      val label = indexedLabel._2
      val animalFileName: String = "/Users/mengpan/Downloads/train/" + fileNumber + ".png"
      val feature = ImageReader.readImageToRGBVector(animalFileName)

      (feature, label)
      }
      .filter(_._1 != None )
      .map(f => Cat(f._1.get, f._2))
      .toList

    new DlCollection[Cat](catList)
 }

  private def getLabels(): Vector[(Int, String)] = {
    Source
      .fromFile("/Users/mengpan/Downloads/trainLabels.csv")
      .getLines()
      .map{eachRow =>
        val split = eachRow.split(",")
        (split(0), split(1))
      }
      .filter{eachRow =>
        eachRow._1 != "id"
      }
      .map{eachRow =>
        (eachRow._1.toInt, eachRow._2)
      }
      .toVector
  }

  private def getBalancedCatNonCatLabels(labels: Vector[(Int, String)]): Vector[(Int, Int)] = {
    labels
      .map{label =>
      val numLabel = label._2 match {
        case "cat" => 1
        case "automobile" => 0
        case _ => 2
      }
      (label._1, numLabel)
    }
      .filter{label =>
        label._2 != 2
      }
  }

}
