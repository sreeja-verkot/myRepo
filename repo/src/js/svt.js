var app = angular.module('myApp',['ngRoute''config','ngTagsInput','simplePagination' ,'ngSanitize','vcRecaptcha','toaster','ngActivityIndicator','ngDialog','EditorDirectives','angular-growl','LocalStorageModule','socialLinks','ngAnimate','ngAutocomplete']);

app.config(['$routeProvider','$locationProvider',
        function($routeProvider, $locationProvider) {
          $routeProvider
          	.when('/svt', {
              templateUrl: 'html/svt.html',
              controller: 'svtConroller'
          	})
            .otherwise({
            	redirectTo: 'svt'
            });

}]);
app.factory("services",function() {


		});

app.controller('svtConroller', function($scope,services, growl, $location, $rootScope,$sce,
		$routeParams,$activityIndicator,$window,ngDialog,toaster) {

        $scope.hloo = function(){
		alert("hlooo");
           }

		});