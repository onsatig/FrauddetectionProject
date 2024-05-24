ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.14"

lazy val root = (project in file("."))
  .settings(
    name := "FrauddetectionProject",
    idePackagePrefix := Some("com.bigdata.frauddectection"),
    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.1",
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.1",
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.1",
      libraryDependencies +="org.apache.spark" %% "spark-avro" % "3.5.1"

)
