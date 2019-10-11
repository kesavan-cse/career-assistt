<?php
include("includes/connectionresume.php");

	if(isset($_POST['post'])){

		$candidate_name = htmlentities(mysqli_real_escape_string($con,$_POST['candidate_name']));
		$qualification = htmlentities(mysqli_real_escape_string($con,$_POST['qualification']));
		$experience = htmlentities(mysqli_real_escape_string($con,$_POST['experience']));
		$salary_expected = htmlentities(mysqli_real_escape_string($con,$_POST['salary_expected']));
		$work_permit = htmlentities(mysqli_real_escape_string($con,$_POST['work_permit']));
		$candidate_image = htmlentities(mysqli_real_escape_string($con,$_POST['candidate_image']));
		$contact_no = htmlentities(mysqli_real_escape_string($con,$_POST['contact_no']));
		$mail = htmlentities(mysqli_real_escape_string($con,$_POST['mail']));
		
		$newgid = sprintf('%05d', rand(0, 999999));



		$check_email = "select * from resume where mail='$mail'";
		$run_email = mysqli_query($check_email);

		$check = mysqli_num_rows($run_email);

		if($check == 1){
			echo "<script>alert('Email already exist, Please try using another email')</script>";
			echo "<script>window.open('resume.php', '_self')</script>";
			exit();
		}

		

		$insert = "insert into resume (candidate_name,qualification,experience,salary_expected,work_permit,candidate_image,contact_no,mail,product_status)
		values('$candidate_name','$qualification','$experience','$salary_expected','$work_permit','$candidate_image','$contact_no','$mail','1')";
		
		$query = mysqli_query($con, $insert);

		if($query){
			echo "<script>alert('Well Done $candidate_name, you are good to go.')</script>";
			echo "<script>window.open('home.php', '_self')</script>";
		}
		else{
			echo "<script>alert('Registration failed, please try again Later!')</script>";
			echo "<script>window.open('home.php', '_self')</script>";
		}
	}
?>