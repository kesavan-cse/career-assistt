<?php
include("includes/connectionpost.php");

	if(isset($_POST['post'])){

		$company_name = htmlentities(mysqli_real_escape_string($con,$_POST['company_name']));
		$company_type = htmlentities(mysqli_real_escape_string($con,$_POST['company_type']));
		$salary = htmlentities(mysqli_real_escape_string($con,$_POST['salary']));
		$rating = htmlentities(mysqli_real_escape_string($con,$_POST['rating']));
		$work_permit = htmlentities(mysqli_real_escape_string($con,$_POST['work_permit']));
		$vacancies = htmlentities(mysqli_real_escape_string($con,$_POST['vacancies']));
		$company_logo = htmlentities(mysqli_real_escape_string($con,$_POST['company_logo']));
		
		$newgid = sprintf('%05d', rand(0, 999999));
		$insert = "insert into product (company_name,company_type,salary,rating,work_permit,vacancies,company_logo,product_quantity,product_status)
		values('$company_name','$company_type','$salary','$rating','$work_permit','$vacancies','$company_logo',10,'1')";
		
		$query = mysqli_query($con, $insert);

		if($query){
			echo "<script>alert('Well Done, you are good to go.')</script>";
			echo "<script>window.open('home.php', '_self')</script>";
		}
		else{
			echo "<script>alert('posting failed, please try again Later!')</script>";
			echo "<script>window.open('home.php', '_self')</script>";
		}
	}
?>