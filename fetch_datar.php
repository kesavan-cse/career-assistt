<?php


include('includes/connectionresume.php');

if(isset($_POST["action"]))
{
	$query = "
		SELECT * FROM resume WHERE product_status = '1'
	";
	if(isset($_POST["minimum_price"], $_POST["maximum_price"]) && !empty($_POST["minimum_price"]) && !empty($_POST["maximum_price"]))
	{
		$query .= "
		 AND salary_expected BETWEEN '".$_POST["minimum_price"]."' AND '".$_POST["maximum_price"]."'
		";
	}
	if(isset($_POST["qualification"]))
	{
		$qualification_filter = implode("','", $_POST["qualification"]);
		$query .= "
		 AND qualification IN('".$qualification_filter."')
		";
	}
	if(isset($_POST["experience"]))
	{
		$experience_filter = implode("','", $_POST["experience"]);
		$query .= "
		 AND experience IN('".$experience_filter."')
		";
	}
	if(isset($_POST["work_permit"]))
	{
		$work_permit_filter = implode("','", $_POST["work_permit"]);
		$query .= "
		 AND work_permit IN('".$work_permit_filter."')
		";
	}
	if(isset($_POST["contact_no"]))
	{
		$contact_no_filter = implode("','", $_POST["contact_no"]);
		$query .= "
		 AND contact_no IN('".$contact_no_filter."')
		";
	}
	if(isset($_POST["mail"]))
	{
		$mail_filter = implode("','", $_POST["mail"]);
		$query .= "
		 AND mail IN('".$mail_filter."')
		";
	}

	$statement = $connect->prepare($query);
	$statement->execute();
	$result = $statement->fetchAll();
	$total_row = $statement->rowCount();
	$output = '';
	if($total_row > 0)
	{
		foreach($result as $row)
		{
			$output .= '
			<div class="col-sm-4 col-lg-3 col-md-3">
				<div style="border:1px solid #ccc; border-radius:5px; padding:16px; margin-bottom:16px; height:450px;">
					<img src="image/'. $row['candidate_image'] .'" alt="" class="img-responsive" >
					<p align="center"><strong><a href="#">'. $row['candidate_name'] .'</a></strong></p>
					<h4 style="text-align:center;" class="text-danger" >'. $row['salary_expected'] .'</h4>
					<p>
					Qualification : '. $row['qualification'] .' <br />
					Experience : '. $row['experience'] .' <br />
					Work permit : '. $row['work_permit'] .'<br />
					Contact no : '. $row['contact_no'] .'<br />
                                                                                                 Mail : '. $row['mail'] .'   </p>
<br><center><button type="sumbit">Recruit!</button></center>

				</div>

			</div>
			';
		}
	}
	else
	{
		$output = '<h3>No Data Found</h3>';
	}
	echo $output;
}

?>