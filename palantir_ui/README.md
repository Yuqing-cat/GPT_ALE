# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm install`
Download all the dependencies for the project

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

## Learn More about UI Development

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

## Learn More about Demo UI Contents

### Login Page
When you the UI, you will be redirected to the login page. Login in with any AAD account and the login page will redirect you to the home page.

If it's the first time you login in to the UI, you will see a permission requested page. That is to get essential permissions for the UI to authenticate your account. Click "Accept" to continue.

### Home Page
A blank page that will be updated in the future.

### Project Page
Shows the model performance of certain project. Click on the legend to change view.
![UI Project Page](../docs/images/ui_project_page.png)

### Active Learning Page
Shows the active learning jobs of certain project. Click on the job name to see the details.
![UI Active Learning Page](../docs/images/ui_active_learning_page.png)

### Active Learning Job Page
Shows the details of certain active learning job. 
Click on the docs to check or annotate the data point.
![UI Active Learning Job Page](../docs/images/ui_active_learning_job_page.png)