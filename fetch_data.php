<?php

//fetch_data.php

include('includes/connectionpost.php');

if(isset($_POST["action"]))
{
	$query = "
		SELECT * FROM product WHERE product_status = '1'
	";
	if(isset($_POST["minimum_price"], $_POST["maximum_price"]) && !empty($_POST["minimum_price"]) && !empty($_POST["maximum_price"]))
	{
		$query .= "
		 AND salary BETWEEN '".$_POST["minimum_price"]."' AND '".$_POST["maximum_price"]."'
		";
	}
	if(isset($_POST["company_type"]))
	{
		$company_type_filter = implode("','", $_POST["company_type"]);
		$query .= "
		 AND company_type IN('".$company_type_filter."')
		";
	}
	if(isset($_POST["rating"]))
	{
		$rating_filter = implode("','", $_POST["rating"]);
		$query .= "
		 AND rating IN('".$rating_filter."')
		";
	}
	if(isset($_POST["work_permit"]))
	{
		$work_permit_filter = implode("','", $_POST["work_permit"]);
		$query .= "
		 AND work_permit IN('".$work_permit_filter."')
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
					<img src="image/'. $row['company_logo'] .'" alt="" class="img-responsive" >
					<p align="center"><strong><a href="#">'. $row['company_name'] .'</a></strong></p>
					<h4 style="text-align:center;" class="text-danger" >'. $row['salary'] .'</h4>
					<p>Vacencies : '. $row['vacancies'].' <br />
					Company type : '. $row['company_type'] .' <br />
					Rating : '. $row['rating'] .' <br />
					Work permit : '. $row['work_permit'] .'  </p><br>
<center><button type="sumbit">Interested!</button></center>
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