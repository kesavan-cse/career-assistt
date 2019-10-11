<?php

//fetch_data.php

include('includes/connectionpost.php');

if(isset($_POST["action"]))
{
	$query = "
		SELECT predicted FROM start WHERE Search_id = '1'
	";

if(isset($_POST["predicted"]))
	{
		$predicted_start = implode("','", $_POST["predicted"]);
		$query .= "
		 AND predicted IN('".$predicted_start."')
		";
	}
$statement = $connect->prepare($query);
	$statement->execute();
	$result = $statement->fetchAll();
	$total_row = $statement->rowCount();
	$output = '';
	if($total_row > 0)
	{
		<h4 style="text-align:center;" class="text-danger" >'. $row['predicted'] .'</h4>
}
else
{
	$output='<h3>No Data Found</h3>
}
echo $output
}
?>