name := "estuary"

version := "1.0"

scalaVersion := "2.12.0"


libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)
libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"

libraryDependencies ++= Seq(
  "com.typesafe.akka" % "akka-actor_2.12" % "2.5.6",
  "com.typesafe.akka" % "akka-remote_2.12" % "2.5.6",
  "com.twitter" % "chill-akka_2.12" % "0.8.4",
  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.slf4j" % "slf4j-log4j12" % "1.7.21"
)


libraryDependencies += "org.scalatest" % "scalatest_2.12" % "3.0.0"

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies ++= Seq(
  "com.chuusai" %% "shapeless" % "2.3.2"
)


// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-core
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1"
