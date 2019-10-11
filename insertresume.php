<!DOCTYPE html>
<html>
<head>
	<title>Post job</title>
	<meta charset="utf-8">
 	<meta name="viewport" content="width=device-width, initial-scale=1">
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
	<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
</head>
<style>
	body{
		overflow-x: hidden;
	}
	.main-content{
		width: 50%;
		height: 40%;
		margin: 10px auto;
		background-color: #fff;
		border: 2px solid #e6e6e6;
		padding: 40px 50px;
	}
	.header{
		border: 0px solid #000;
		margin-bottom: 5px;
	}
	.well{
		background-color: #187FAB;
	}
	#signup{
		width: 60%;
		border-radius: 30px;
	}

</style>
<body>
<body background="bg.jpg">
<div class="row">
	<div class="col-sm-12">
		<div class="well">
			<marquee><h1 style="color: white;">Carrier Assist</h1></marquee>
		</div>
	</div>
</div>
<div class="row">
<body background="bg.jpg">
	<div class="col-sm-12">
		<div class="main-content">
			<div class="header">
				<h3 style="text-align: center;"><strong>fill your details</strong></h3>
				<hr>
			</div>
			<div class="l-part">
				<form action="" method="post">
					<div class="input-group">
						<span class="input-group-addon"><i class="glyphicon glyphicon-user"></i></span>
						<input type="text" class="form-control" placeholder="Name" name="candidate_name" required="required">
					</div><br>
					<div class="input-group">
						<span class="input-group-addon"><i class="glyphicon glyphicon-pencil"></i></span>
						<input type="text" class="form-control" placeholder="qualification" name="qualification" required="required">
					</div><br>
					<div class="input-group">
					<span class="input-group-addon"><i class="glyphicon glyphicon-pencil"></i></span>
						<input type="number" class="form-control" placeholder="experience" name="experience" required="required">
				</div><br>
				<div class="input-group">
					<span class="input-group-addon"><i class="glyphicon glyphicon-pencil"></i></span>
						<input type="number" class="form-control" placeholder="salary_expected" name="salary_expected" required="required">
				</div><br>
					
					<div class="input-group">
						<span class="input-group-addon"><i class="glyphicon glyphicon-chevron-down"></i></span>
						<select class="form-control" name="work_permit" required="required">
							<option disabled>Select your option</option>
							<option>work permit</option>
							<option>Full time</option>
							<option>Part time</option>
							
						</select>
					</div><br>
					
					<div class="input-group">
						<span class="input-group-addon"><i class="glyphicon glyphicon-picture"></i></span>
						<h6>Company logo</h6>  
                                                                                       <input type="file" class="form-control" placeholder="Company Logo" name="company_logo" required="required">
					
			 
			  
	        		                                   </div><br>
					<div class="input-group">
						<span class="input-group-addon"><i class="glyphicon glyphicon-pencil"></i></span>
						<input type="text" class="form-control" placeholder="contact_no" name="contact_no" required="required">
					</div><br>
					<div class="input-group">
						<span class="input-group-addon"><i class="glyphicon glyphicon-pencil"></i></span>
						<input type="text" class="form-control" placeholder="E-mail" name="mail" required="required">
					</div><br>
		    
	  		
					
					<center><button id="post" class="btn btn-info btn-lg" name="post">POST</button></center>
					<?php include("insert_resume.php"); ?>
				</form>
			</div>
		</div>
	</div>
</div>
</body>
</html>