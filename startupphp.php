<?php
include("includes/connectionstartup.php");

	if(isset($_POST['post'])){

		$Crop_type = htmlentities(mysqli_real_escape_string($con,$_POST['Crop_type']));
		$Expected_Year = htmlentities(mysqli_real_escape_string($con,$_POST['Expected_Year']));
		$District = htmlentities(mysqli_real_escape_string($con,$_POST['District']));
		
		
		$insert = "replace into start (Search_id,Crop_type,Expected_Year,District)
		values('1','$Crop_type','$Expected_Year','$District')";
		
		$query = mysqli_query($con, $insert);

		if($query){
			echo "<script>alert('Well Done , you are good to go.')</script>";
			echo "<script>window.open('startup.php', '_self')</script>";
		}
		else{
			echo "<script>alert('Search failed, please try again Later!')</script>";
			echo "<script>window.open('startup.php', '_self')</script>";
		}
	}
?>