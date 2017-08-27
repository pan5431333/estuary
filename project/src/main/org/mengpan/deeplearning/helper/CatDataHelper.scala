package org.mengpan.deeplearning.helper

import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.data.Cat

import scala.io.Source

/**
  * Created by mengpan on 2017/8/15.
  */
object CatDataHelper {
  val logger = Logger.getLogger("CatDataHelper")


  def getAllCatData: DlCollection[Cat] = {

    val labels = getLabels

    val catNonCatLabels = getBalancedBatNonCatLabels(labels)

    val catList = catNonCatLabels.map{indexedLabel =>

      val fileNumber = indexedLabel._1
      val label = indexedLabel._2
      val animalFileName: String = "/Users/mengpan/Downloads/train/" + fileNumber + ".png"
      val feature = getFeatureForOneAnimal(animalFileName)

      feature match {
        case Some(s) => Cat(s, label)
        case None => Cat(DenseVector.zeros[Double](10), label)
      }
    }
      .filter{cat =>
        cat.feature.length != 10
      }
      .toList

    new DlCollection[Cat](catList)
 }

  private def getFeatureForOneAnimal(animalFileName: String): Option[DenseVector[Double]] = {
    logger.info("Reading file: " + animalFileName)

    try {
      val image = ImageIO.read(new File(animalFileName))
      val imageData = image.getData

      val redVector = DenseVector.zeros[Double](imageData.getHeight * imageData.getWidth)
      val greenVector = DenseVector.zeros[Double](imageData.getHeight * imageData.getWidth)
      val blueVector = DenseVector.zeros[Double](imageData.getHeight * imageData.getWidth)

      (0 until imageData.getHeight).foreach{height =>
        (0 until imageData.getWidth).foreach{width =>
          val RGB = imageData.getPixel(width, height, Array(0, 0, 0))
          redVector(width + height*10) = RGB(0)
          greenVector(width + height*10) = RGB(1)
          blueVector(width + height*10) = RGB(2)
        }
      }

      val resVector = DenseMatrix(redVector, greenVector, blueVector).reshape(imageData.getHeight*imageData.getWidth*3, 1).toDenseVector
      Some(resVector)
    } catch {
      case _: Exception => None
    }
  }

  private def getLabels: Vector[(Int, String)] = {
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

  private def getBalancedBatNonCatLabels(labels: Vector[(Int, String)]): Vector[(Int, Int)] = {
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
